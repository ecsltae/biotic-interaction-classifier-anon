#!/usr/bin/env python3
"""
Relation Extraction Model for Host-Pathogen Interactions.

This module implements a joint entity and relation extraction model
based on BERT with span classification, inspired by SpERT architecture.

Architecture:
1. BERT encoder for contextual representations
2. Span classifier for entity detection
3. Relation classifier for entity pairs

Mathematical Foundation:
- Entity span representation: h_span = [h_start; h_end; width_embedding]
- Relation representation: h_rel = [h_span1; h_span2; context_pooling]
- Classification: softmax(W * h + b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# Entity and relation type definitions
ENTITY_TYPES = ['O', 'HOST', 'PATHOGEN', 'VECTOR', 'RESERVOIR', 'DISEASE']
RELATION_TYPES = [
    'NO_RELATION',
    'INFECTED_BY', 'INFECTS',
    'TRANSMITS', 'TRANSMITTED_BY',
    'VECTOR_OF', 'HAS_VECTOR',
    'RESERVOIR_FOR', 'HAS_RESERVOIR',
    'SUSCEPTIBLE_TO', 'RESISTANT_TO',
    'COLONIZED_BY', 'COLONIZES',
    'CAUSES_DISEASE', 'DISEASE_CAUSED_BY',
    'CO_INFECTS_WITH'
]


@dataclass
class SpanPrediction:
    """Predicted entity span."""
    start: int
    end: int
    entity_type: str
    score: float


@dataclass
class RelationPrediction:
    """Predicted relation."""
    head_span: Tuple[int, int]
    tail_span: Tuple[int, int]
    relation_type: str
    score: float


class SpanClassifier(nn.Module):
    """
    Span-based entity classifier.

    For each candidate span (i, j), computes:
    h_span = [h_i; h_j; width_emb(j-i)]
    p(entity_type) = softmax(W * h_span + b)
    """

    def __init__(self, hidden_size: int, num_entity_types: int, max_span_width: int = 10):
        super().__init__()
        self.max_span_width = max_span_width

        # Width embeddings (learnable)
        self.width_embedding = nn.Embedding(max_span_width, hidden_size // 4)

        # Span representation: [start; end; width]
        span_size = hidden_size * 2 + hidden_size // 4

        self.classifier = nn.Sequential(
            nn.Linear(span_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_entity_types)
        )

    def forward(self, hidden_states: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            spans: (batch, num_spans, 2) - start and end indices

        Returns:
            span_logits: (batch, num_spans, num_entity_types)
        """
        batch_size, num_spans, _ = spans.shape

        # Get start and end representations
        start_indices = spans[:, :, 0]  # (batch, num_spans)
        end_indices = spans[:, :, 1]    # (batch, num_spans)

        # Gather hidden states
        # Expand indices for gathering
        start_expanded = start_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        end_expanded = end_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))

        start_repr = torch.gather(hidden_states, 1, start_expanded)  # (batch, num_spans, hidden)
        end_repr = torch.gather(hidden_states, 1, end_expanded)      # (batch, num_spans, hidden)

        # Width embeddings
        widths = torch.clamp(end_indices - start_indices, 0, self.max_span_width - 1)
        width_repr = self.width_embedding(widths)  # (batch, num_spans, hidden//4)

        # Concatenate span representation
        span_repr = torch.cat([start_repr, end_repr, width_repr], dim=-1)

        # Classify
        logits = self.classifier(span_repr)

        return logits


class RelationClassifier(nn.Module):
    """
    Relation classifier for entity pairs.

    For entity pair (e1, e2), computes:
    h_rel = [h_e1; h_e2; context_between]
    p(relation_type) = softmax(W * h_rel + b)
    """

    def __init__(self, hidden_size: int, num_relation_types: int):
        super().__init__()

        # Relation representation: [span1; span2; context]
        rel_size = hidden_size * 3

        self.classifier = nn.Sequential(
            nn.Linear(rel_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_relation_types)
        )

    def forward(
        self,
        span1_repr: torch.Tensor,
        span2_repr: torch.Tensor,
        context_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            span1_repr: (batch, num_pairs, hidden_size)
            span2_repr: (batch, num_pairs, hidden_size)
            context_repr: (batch, num_pairs, hidden_size)

        Returns:
            rel_logits: (batch, num_pairs, num_relation_types)
        """
        rel_repr = torch.cat([span1_repr, span2_repr, context_repr], dim=-1)
        logits = self.classifier(rel_repr)
        return logits


class HostPathogenRelationExtractor(nn.Module):
    """
    Joint Entity and Relation Extraction Model for Host-Pathogen Interactions.

    Based on SpERT architecture with modifications for biomedical domain.
    Uses BiomedBERT as the encoder.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        max_span_width: int = 10,
        entity_types: List[str] = ENTITY_TYPES,
        relation_types: List[str] = RELATION_TYPES,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.entity_types = entity_types
        self.relation_types = relation_types
        self.max_span_width = max_span_width

        # BERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Span classifier for entities
        self.span_classifier = SpanClassifier(
            hidden_size=hidden_size,
            num_entity_types=len(entity_types),
            max_span_width=max_span_width
        )

        # Relation classifier
        self.relation_classifier = RelationClassifier(
            hidden_size=hidden_size,
            num_relation_types=len(relation_types)
        )

        # Context pooling for relation classification
        self.context_pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input text with BERT."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def generate_spans(self, seq_len: int, max_span_width: int) -> List[Tuple[int, int]]:
        """Generate all possible spans up to max_span_width."""
        spans = []
        for start in range(seq_len):
            for end in range(start + 1, min(start + max_span_width + 1, seq_len + 1)):
                spans.append((start, end))
        return spans

    def get_span_representation(
        self,
        hidden_states: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor
    ) -> torch.Tensor:
        """Get pooled representation for spans."""
        batch_size = hidden_states.size(0)
        hidden_size = hidden_states.size(-1)

        # Simple pooling: average of start and end
        start_expanded = span_starts.unsqueeze(-1).expand(-1, -1, hidden_size)
        end_expanded = (span_ends - 1).clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden_size)

        start_repr = torch.gather(hidden_states, 1, start_expanded)
        end_repr = torch.gather(hidden_states, 1, end_expanded)

        return (start_repr + end_repr) / 2

    def get_context_representation(
        self,
        hidden_states: torch.Tensor,
        span1_ends: torch.Tensor,
        span2_starts: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get context representation between two spans."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_pairs = span1_ends.size(1)

        context_reprs = []
        for b in range(batch_size):
            batch_contexts = []
            for p in range(num_pairs):
                start = span1_ends[b, p].item()
                end = span2_starts[b, p].item()

                if start < end:
                    # Context between spans
                    context = hidden_states[b, start:end].mean(dim=0)
                else:
                    # Spans overlap or adjacent, use CLS token
                    context = hidden_states[b, 0]

                batch_contexts.append(context)

            context_reprs.append(torch.stack(batch_contexts))

        context_repr = torch.stack(context_reprs)
        return self.context_pooler(context_repr)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
        relation_pairs: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            spans: (batch, num_spans, 2) - candidate spans [start, end)
            span_labels: (batch, num_spans) - entity type labels
            relation_pairs: (batch, num_pairs, 2) - pairs of span indices
            relation_labels: (batch, num_pairs) - relation type labels

        Returns:
            Dictionary with logits and optional losses
        """
        # Encode input
        hidden_states = self.encode(input_ids, attention_mask)

        outputs = {}

        # Entity classification
        if spans is not None:
            span_logits = self.span_classifier(hidden_states, spans)
            outputs['span_logits'] = span_logits

            if span_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                span_loss = loss_fct(
                    span_logits.view(-1, len(self.entity_types)),
                    span_labels.view(-1)
                )
                outputs['span_loss'] = span_loss

        # Relation classification
        if relation_pairs is not None and spans is not None:
            batch_size = hidden_states.size(0)
            num_pairs = relation_pairs.size(1)

            # Get span indices for each relation pair
            span1_indices = relation_pairs[:, :, 0]  # (batch, num_pairs)
            span2_indices = relation_pairs[:, :, 1]  # (batch, num_pairs)

            # Get span boundaries
            span1_starts = torch.gather(spans[:, :, 0], 1, span1_indices)
            span1_ends = torch.gather(spans[:, :, 1], 1, span1_indices)
            span2_starts = torch.gather(spans[:, :, 0], 1, span2_indices)
            span2_ends = torch.gather(spans[:, :, 1], 1, span2_indices)

            # Get span representations
            span1_repr = self.get_span_representation(hidden_states, span1_starts, span1_ends)
            span2_repr = self.get_span_representation(hidden_states, span2_starts, span2_ends)

            # Get context representation
            context_repr = self.get_context_representation(
                hidden_states, span1_ends, span2_starts, attention_mask
            )

            # Classify relations
            rel_logits = self.relation_classifier(span1_repr, span2_repr, context_repr)
            outputs['relation_logits'] = rel_logits

            if relation_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                rel_loss = loss_fct(
                    rel_logits.view(-1, len(self.relation_types)),
                    relation_labels.view(-1)
                )
                outputs['relation_loss'] = rel_loss

        # Combined loss
        if 'span_loss' in outputs and 'relation_loss' in outputs:
            outputs['loss'] = outputs['span_loss'] + outputs['relation_loss']
        elif 'span_loss' in outputs:
            outputs['loss'] = outputs['span_loss']
        elif 'relation_loss' in outputs:
            outputs['loss'] = outputs['relation_loss']

        return outputs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_threshold: float = 0.5,
        relation_threshold: float = 0.5
    ) -> Tuple[List[SpanPrediction], List[RelationPrediction]]:
        """
        Predict entities and relations.

        Args:
            input_ids: (1, seq_len) - single example
            attention_mask: (1, seq_len)
            span_threshold: confidence threshold for entity prediction
            relation_threshold: confidence threshold for relation prediction

        Returns:
            entities: list of SpanPrediction
            relations: list of RelationPrediction
        """
        self.eval()
        with torch.no_grad():
            seq_len = attention_mask.sum().item()

            # Generate all candidate spans
            span_list = self.generate_spans(seq_len, self.max_span_width)
            spans = torch.tensor([[s for s in span_list]], device=input_ids.device)

            # Get span logits
            hidden_states = self.encode(input_ids, attention_mask)
            span_logits = self.span_classifier(hidden_states, spans)
            span_probs = F.softmax(span_logits, dim=-1)

            # Filter entities (non-O predictions above threshold)
            entities = []
            entity_spans = []
            for i, (start, end) in enumerate(span_list):
                probs = span_probs[0, i]
                pred_type = probs.argmax().item()

                if pred_type > 0 and probs[pred_type] > span_threshold:  # Not 'O'
                    entities.append(SpanPrediction(
                        start=start,
                        end=end,
                        entity_type=self.entity_types[pred_type],
                        score=probs[pred_type].item()
                    ))
                    entity_spans.append((i, start, end))

            # Predict relations between entity pairs
            relations = []
            if len(entity_spans) >= 2:
                pairs = []
                pair_indices = []
                for i, (idx1, s1, e1) in enumerate(entity_spans):
                    for j, (idx2, s2, e2) in enumerate(entity_spans):
                        if i != j:
                            pairs.append((s1, e1, s2, e2))
                            pair_indices.append((idx1, idx2))

                if pairs:
                    # Create relation pairs tensor
                    relation_pairs = torch.tensor([[p for p in pair_indices]], device=input_ids.device)

                    # Get relation predictions
                    outputs = self.forward(
                        input_ids, attention_mask,
                        spans=spans,
                        relation_pairs=relation_pairs
                    )

                    rel_probs = F.softmax(outputs['relation_logits'], dim=-1)

                    for k, (s1, e1, s2, e2) in enumerate(pairs):
                        probs = rel_probs[0, k]
                        pred_type = probs.argmax().item()

                        if pred_type > 0 and probs[pred_type] > relation_threshold:  # Not 'NO_RELATION'
                            relations.append(RelationPrediction(
                                head_span=(s1, e1),
                                tail_span=(s2, e2),
                                relation_type=self.relation_types[pred_type],
                                score=probs[pred_type].item()
                            ))

            return entities, relations


def create_model(model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
    """Factory function to create the relation extraction model."""
    return HostPathogenRelationExtractor(model_name=model_name)


if __name__ == "__main__":
    # Test model creation
    print("Creating Host-Pathogen Relation Extractor...")
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Entity types: {ENTITY_TYPES}")
    print(f"Relation types: {RELATION_TYPES}")
