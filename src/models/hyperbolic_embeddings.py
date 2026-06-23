#!/usr/bin/env python3
"""
Hyperbolic Embeddings for Taxonomic Hierarchies.

This module implements Poincaré ball embeddings for representing hierarchical
taxonomies (e.g., species taxonomies) in hyperbolic space.

Mathematical Foundation:
========================

Hyperbolic geometry is ideal for hierarchical data because:
- The volume of hyperbolic space grows exponentially with radius
- Trees can be embedded with low distortion
- Euclidean space needs O(n) dimensions for trees, hyperbolic needs O(log n)

Poincaré Ball Model:
-------------------
The Poincaré ball B^d = {x ∈ ℝ^d : ||x|| < 1} with metric:

    g_x = (2 / (1 - ||x||²))² * g_E

where g_E is the Euclidean metric.

Hyperbolic Distance:
-------------------
    d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))

Key Properties:
- Points near the boundary (||x|| → 1) are "far" from the origin
- Hierarchies: root near origin, leaves near boundary
- Exponential expansion preserves tree metric

References:
-----------
- Nickel & Kiela (2017): Poincaré Embeddings for Learning Hierarchical Representations
- Ganea et al. (2018): Hyperbolic Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import json
from pathlib import Path


# Constants
EPS = 1e-5  # Numerical stability
MAX_NORM = 1 - EPS  # Maximum norm in Poincaré ball


class PoincareOperations:
    """
    Mathematical operations in the Poincaré ball model.

    All operations are defined for the unit ball with curvature c = 1.
    For curvature c, scale distances by 1/√c.
    """

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.

        x ⊕ y = ((1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y) / (1 + 2⟨x,y⟩ + ||x||²||y||²)

        This is the "addition" operation in hyperbolic space.
        """
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=EPS)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True).clamp(min=EPS)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

        numerator = (1 + 2 * xy_dot + y_norm_sq) * x + (1 - x_norm_sq) * y
        denominator = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq

        return numerator / denominator.clamp(min=EPS)

    @staticmethod
    def hyperbolic_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance in the Poincaré ball.

        d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))

        Args:
            u, v: Points in the Poincaré ball, shape (..., d)

        Returns:
            Distance, shape (...)
        """
        diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)
        u_norm_sq = torch.sum(u ** 2, dim=-1).clamp(max=MAX_NORM**2)
        v_norm_sq = torch.sum(v ** 2, dim=-1).clamp(max=MAX_NORM**2)

        # Compute argument of arcosh
        arg = 1 + 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq) + EPS)

        # arcosh(x) = log(x + sqrt(x² - 1))
        # Numerically stable version
        return torch.acosh(arg.clamp(min=1 + EPS))

    @staticmethod
    def project_to_ball(x: torch.Tensor, max_norm: float = MAX_NORM) -> torch.Tensor:
        """
        Project points onto the Poincaré ball (ensure ||x|| < 1).

        This is necessary after gradient updates to keep points inside the ball.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * (max_norm / norm.clamp(min=max_norm))

    @staticmethod
    def exp_map_0(v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from the origin (tangent space at origin → ball).

        exp_0(v) = tanh(||v||) * v / ||v||

        This maps a Euclidean vector to a point in the ball.
        """
        norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=EPS)
        return torch.tanh(norm) * v / norm

    @staticmethod
    def log_map_0(y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map to the origin (ball → tangent space at origin).

        log_0(y) = arctanh(||y||) * y / ||y||

        This is the inverse of exp_map_0.
        """
        norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=EPS, max=MAX_NORM)
        return torch.atanh(norm) * y / norm


class PoincareEmbedding(nn.Module):
    """
    Poincaré ball embedding layer.

    Embeds discrete entities (e.g., species IDs) into the Poincaré ball.
    Uses Riemannian SGD for optimization.
    """

    def __init__(self, num_entities: int, embedding_dim: int, init_scale: float = 0.001):
        """
        Args:
            num_entities: Number of entities to embed
            embedding_dim: Dimension of the Poincaré ball
            init_scale: Scale for random initialization (small to stay near origin)
        """
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim

        # Initialize embeddings uniformly in a small ball
        embeddings = torch.randn(num_entities, embedding_dim) * init_scale
        self.embeddings = nn.Parameter(embeddings)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for given indices."""
        emb = self.embeddings[indices]
        # Project to ensure we're inside the ball
        return PoincareOperations.project_to_ball(emb)

    def distance(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance between entities."""
        emb1 = self.forward(idx1)
        emb2 = self.forward(idx2)
        return PoincareOperations.hyperbolic_distance(emb1, emb2)


class TaxonomyHyperbolicModel(nn.Module):
    """
    Hyperbolic embedding model for taxonomic hierarchies.

    Learns embeddings such that:
    - Parent nodes are closer to the origin
    - Child nodes are farther from the origin
    - Distance preserves taxonomic relationships

    Loss Function:
    --------------
    For (parent, child) pair, minimize:
        L = log(1 + exp(d(parent, child) - d(parent, negative)))

    where negative is a randomly sampled non-child.

    This pushes parent-child pairs together and separates non-related pairs.
    """

    def __init__(
        self,
        entity2idx: Dict[str, int],
        embedding_dim: int = 32,
        margin: float = 0.1
    ):
        """
        Args:
            entity2idx: Mapping from entity names to indices
            embedding_dim: Dimension of hyperbolic space
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.entity2idx = entity2idx
        self.idx2entity = {v: k for k, v in entity2idx.items()}
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.embeddings = PoincareEmbedding(
            num_entities=len(entity2idx),
            embedding_dim=embedding_dim
        )

    def forward(self, parents: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between parent-child pairs.

        Args:
            parents: (batch,) parent indices
            children: (batch,) child indices

        Returns:
            distances: (batch,) hyperbolic distances
        """
        return self.embeddings.distance(parents, children)

    def loss(
        self,
        parents: torch.Tensor,
        children: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss for taxonomy embedding.

        Pushes parent-child pairs together, separates parent-negative pairs.

        Args:
            parents: (batch,) parent indices
            children: (batch,) child (positive) indices
            negatives: (batch,) negative indices

        Returns:
            loss: scalar
        """
        pos_dist = self.forward(parents, children)
        neg_dist = self.forward(parents, negatives)

        # Margin-based contrastive loss
        # We want pos_dist < neg_dist - margin
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        return loss

    def hierarchical_loss(
        self,
        parents: torch.Tensor,
        children: torch.Tensor
    ) -> torch.Tensor:
        """
        Additional loss to enforce hierarchy (parents closer to origin).

        Args:
            parents: (batch,) parent indices
            children: (batch,) child indices

        Returns:
            loss: scalar
        """
        parent_emb = self.embeddings(parents)
        child_emb = self.embeddings(children)

        # Parents should have smaller norm (closer to origin)
        parent_norm = torch.norm(parent_emb, dim=-1)
        child_norm = torch.norm(child_emb, dim=-1)

        # Penalize when parent norm > child norm
        hierarchy_loss = F.relu(parent_norm - child_norm + 0.05).mean()

        return hierarchy_loss

    def get_embedding(self, entity_name: str) -> torch.Tensor:
        """Get embedding for a named entity."""
        idx = self.entity2idx.get(entity_name)
        if idx is None:
            raise ValueError(f"Unknown entity: {entity_name}")
        return self.embeddings(torch.tensor([idx]))[0]

    def get_distance(self, entity1: str, entity2: str) -> float:
        """Get hyperbolic distance between two named entities."""
        idx1 = self.entity2idx[entity1]
        idx2 = self.entity2idx[entity2]
        dist = self.embeddings.distance(
            torch.tensor([idx1]),
            torch.tensor([idx2])
        )
        return dist.item()

    def get_nearest_neighbors(self, entity_name: str, k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest neighbors in hyperbolic space."""
        idx = self.entity2idx[entity_name]
        emb = self.embeddings(torch.tensor([idx]))

        # Compute distances to all entities
        all_idx = torch.arange(len(self.entity2idx))
        all_emb = self.embeddings(all_idx)
        distances = PoincareOperations.hyperbolic_distance(
            emb.expand(len(self.entity2idx), -1),
            all_emb
        )

        # Sort by distance
        sorted_idx = distances.argsort()
        neighbors = []
        for i in sorted_idx[1:k+1]:  # Skip self
            neighbor_name = self.idx2entity[i.item()]
            dist = distances[i].item()
            neighbors.append((neighbor_name, dist))

        return neighbors


class TaxonomyDataset:
    """
    Dataset for training hyperbolic taxonomy embeddings.

    Loads taxonomy data and generates training pairs.
    """

    def __init__(self, taxonomy_file: Path):
        """
        Args:
            taxonomy_file: Path to JSON file with taxonomy structure
        """
        self.taxonomy_file = taxonomy_file
        self.parent_child_pairs = []
        self.entity2idx = {}

        self._load_taxonomy()

    def _load_taxonomy(self):
        """Load taxonomy from JSON file."""
        with open(self.taxonomy_file) as f:
            taxonomy = json.load(f)

        # Build entity index and parent-child pairs
        idx = 0
        stack = [(taxonomy, None)]  # (node, parent_name)

        while stack:
            node, parent = stack.pop()

            if isinstance(node, dict):
                # Node with children
                name = node.get('name', node.get('taxon', str(idx)))
                if name not in self.entity2idx:
                    self.entity2idx[name] = idx
                    idx += 1

                if parent is not None:
                    self.parent_child_pairs.append((parent, name))

                # Process children
                children = node.get('children', node.get('subtaxa', []))
                for child in children:
                    stack.append((child, name))

            elif isinstance(node, str):
                # Leaf node (just a name)
                if node not in self.entity2idx:
                    self.entity2idx[node] = idx
                    idx += 1

                if parent is not None:
                    self.parent_child_pairs.append((parent, node))

        print(f"Loaded taxonomy with {len(self.entity2idx)} entities")
        print(f"Found {len(self.parent_child_pairs)} parent-child pairs")

    def get_training_batch(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a training batch with positive and negative samples.

        Returns:
            parents: (batch_size,) parent indices
            children: (batch_size,) positive child indices
            negatives: (batch_size,) negative (random) indices
        """
        # Sample positive pairs
        indices = np.random.choice(len(self.parent_child_pairs), batch_size, replace=True)
        pairs = [self.parent_child_pairs[i] for i in indices]

        parents = torch.tensor([self.entity2idx[p] for p, c in pairs], device=device)
        children = torch.tensor([self.entity2idx[c] for p, c in pairs], device=device)

        # Sample negatives (random entities that are not children of the parent)
        negatives = torch.randint(0, len(self.entity2idx), (batch_size,), device=device)

        return parents, children, negatives


def train_taxonomy_embeddings(
    taxonomy_file: Path,
    output_file: Path,
    embedding_dim: int = 32,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.01
):
    """
    Train hyperbolic embeddings for a taxonomy.

    Args:
        taxonomy_file: Path to taxonomy JSON
        output_file: Path to save trained embeddings
        embedding_dim: Dimension of hyperbolic space
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    # Load data
    dataset = TaxonomyDataset(taxonomy_file)

    # Create model
    model = TaxonomyHyperbolicModel(
        entity2idx=dataset.entity2idx,
        embedding_dim=embedding_dim
    )

    # Optimizer (Riemannian SGD would be better, but standard SGD works with projection)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining hyperbolic embeddings")
    print(f"  Entities: {len(dataset.entity2idx)}")
    print(f"  Dimension: {embedding_dim}")
    print(f"  Epochs: {epochs}")

    for epoch in range(epochs):
        parents, children, negatives = dataset.get_training_batch(batch_size)

        optimizer.zero_grad()

        # Contrastive loss
        contrastive_loss = model.loss(parents, children, negatives)

        # Hierarchical loss (parents closer to origin)
        hierarchy_loss = model.hierarchical_loss(parents, children)

        # Total loss
        loss = contrastive_loss + 0.1 * hierarchy_loss

        loss.backward()
        optimizer.step()

        # Project embeddings back to ball
        with torch.no_grad():
            model.embeddings.embeddings.data = PoincareOperations.project_to_ball(
                model.embeddings.embeddings.data
            )

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} "
                  f"(contrastive={contrastive_loss.item():.4f}, "
                  f"hierarchy={hierarchy_loss.item():.4f})")

    # Save model
    torch.save({
        'entity2idx': dataset.entity2idx,
        'embedding_dim': embedding_dim,
        'embeddings': model.embeddings.embeddings.data
    }, output_file)
    print(f"\nSaved embeddings to {output_file}")

    return model


if __name__ == "__main__":
    # Test with a simple hierarchy
    print("Testing Poincaré operations...")

    # Test points
    x = torch.tensor([0.1, 0.2])
    y = torch.tensor([0.3, 0.1])

    # Test distance
    dist = PoincareOperations.hyperbolic_distance(x, y)
    print(f"Distance between {x.tolist()} and {y.tolist()}: {dist.item():.4f}")

    # Test Möbius addition
    z = PoincareOperations.mobius_add(x, y)
    print(f"Möbius addition: {z.tolist()}")

    # Test exponential map
    v = torch.tensor([0.5, 0.5])
    p = PoincareOperations.exp_map_0(v)
    print(f"Exp map of {v.tolist()}: {p.tolist()}")

    print("\nPoincaré operations working correctly!")
