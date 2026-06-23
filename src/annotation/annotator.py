#!/usr/bin/env python3
"""
Simple annotation tool for host-pathogen interaction sentences.

This tool allows manual annotation of entities and relations in sentences
for training relation extraction models.

Usage:
    python annotator.py --input sentences.csv --output annotations.jsonl
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse


@dataclass
class Entity:
    """Represents an annotated entity."""
    id: int
    text: str
    type: str  # HOST, PATHOGEN, VECTOR, DISEASE
    start: int
    end: int
    ncbi_taxid: Optional[str] = None


@dataclass
class Relation:
    """Represents a relation between entities."""
    head: int  # Entity ID
    relation: str
    tail: int  # Entity ID


@dataclass
class Annotation:
    """Complete annotation for a sentence."""
    sentence_id: str
    text: str
    entities: List[Entity]
    relations: List[Relation]
    annotator: str
    notes: str = ""


# Valid entity types
ENTITY_TYPES = ['HOST', 'PATHOGEN', 'VECTOR', 'RESERVOIR', 'DISEASE']

# Valid relation types
RELATION_TYPES = [
    'INFECTED_BY', 'INFECTS',
    'TRANSMITS', 'TRANSMITTED_BY',
    'VECTOR_OF', 'HAS_VECTOR',
    'RESERVOIR_FOR', 'HAS_RESERVOIR',
    'SUSCEPTIBLE_TO', 'RESISTANT_TO',
    'COLONIZED_BY', 'COLONIZES',
    'CAUSES_DISEASE', 'DISEASE_CAUSED_BY',
    'CO_INFECTS_WITH', 'NO_RELATION'
]


class AnnotationTool:
    """Terminal-based annotation tool for host-pathogen sentences."""

    def __init__(self, input_file: Path, output_file: Path, annotator: str = "anonymous"):
        self.input_file = input_file
        self.output_file = output_file
        self.annotator = annotator
        self.annotations: List[Annotation] = []

        # Load existing annotations if file exists
        if output_file.exists():
            self._load_annotations()

    def _load_annotations(self):
        """Load existing annotations from JSONL file."""
        with open(self.output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                entities = [Entity(**e) for e in data['entities']]
                relations = [Relation(**r) for r in data['relations']]
                ann = Annotation(
                    sentence_id=data['sentence_id'],
                    text=data['text'],
                    entities=entities,
                    relations=relations,
                    annotator=data.get('annotator', 'unknown'),
                    notes=data.get('notes', '')
                )
                self.annotations.append(ann)

    def _save_annotation(self, ann: Annotation):
        """Append annotation to JSONL file."""
        data = {
            'sentence_id': ann.sentence_id,
            'text': ann.text,
            'entities': [asdict(e) for e in ann.entities],
            'relations': [asdict(r) for r in ann.relations],
            'annotator': ann.annotator,
            'notes': ann.notes
        }
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _highlight_entities(self, text: str, entities: List[Entity]) -> str:
        """Highlight entities in text with markers."""
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        highlighted = text
        for ent in sorted_entities:
            marker = f"[{ent.id}:{ent.type}]"
            highlighted = (
                highlighted[:ent.end] +
                f"]{marker}" +
                highlighted[ent.end:]
            )
            highlighted = (
                highlighted[:ent.start] +
                "[" +
                highlighted[ent.start:]
            )

        return highlighted

    def _find_span(self, text: str, mention: str) -> Tuple[int, int]:
        """Find character span of mention in text."""
        # Case-insensitive search
        match = re.search(re.escape(mention), text, re.IGNORECASE)
        if match:
            return match.start(), match.end()
        return -1, -1

    def annotate_entities(self, text: str) -> List[Entity]:
        """Interactive entity annotation."""
        print("\n" + "="*60)
        print("ENTITY ANNOTATION")
        print("="*60)
        print(f"\nSentence: {text}")
        print("\nEntity types:", ", ".join(ENTITY_TYPES))
        print("Commands: 'done' to finish, 'clear' to start over")

        entities = []
        entity_id = 0

        while True:
            if entities:
                print(f"\nCurrent entities:")
                for e in entities:
                    print(f"  [{e.id}] {e.text} ({e.type})")

            mention = input("\nEnter entity text (or 'done'): ").strip()

            if mention.lower() == 'done':
                break
            if mention.lower() == 'clear':
                entities = []
                entity_id = 0
                continue

            # Find span in text
            start, end = self._find_span(text, mention)
            if start == -1:
                print(f"  Warning: '{mention}' not found in text. Trying exact match...")
                # Try finding substring
                if mention.lower() in text.lower():
                    idx = text.lower().index(mention.lower())
                    start, end = idx, idx + len(mention)
                    mention = text[start:end]  # Get exact case from text
                else:
                    print(f"  Error: Could not find '{mention}' in text")
                    continue

            # Get entity type
            print(f"  Entity types: {', '.join([f'{i}:{t}' for i, t in enumerate(ENTITY_TYPES)])}")
            type_input = input("  Enter type (number or name): ").strip()

            try:
                if type_input.isdigit():
                    entity_type = ENTITY_TYPES[int(type_input)]
                else:
                    entity_type = type_input.upper()
                    if entity_type not in ENTITY_TYPES:
                        print(f"  Invalid type. Choose from: {ENTITY_TYPES}")
                        continue
            except (IndexError, ValueError):
                print(f"  Invalid type. Choose from: {ENTITY_TYPES}")
                continue

            entity = Entity(
                id=entity_id,
                text=mention,
                type=entity_type,
                start=start,
                end=end
            )
            entities.append(entity)
            entity_id += 1
            print(f"  Added: [{entity.id}] {entity.text} ({entity.type}) at {start}-{end}")

        return entities

    def annotate_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Interactive relation annotation."""
        if len(entities) < 2:
            print("\nNeed at least 2 entities to annotate relations.")
            return []

        print("\n" + "="*60)
        print("RELATION ANNOTATION")
        print("="*60)
        print(f"\nSentence: {text}")
        print(f"\nHighlighted: {self._highlight_entities(text, entities)}")
        print("\nEntities:")
        for e in entities:
            print(f"  [{e.id}] {e.text} ({e.type})")
        print("\nRelation types:")
        for i, rel in enumerate(RELATION_TYPES):
            print(f"  {i}: {rel}")
        print("\nCommands: 'done' to finish, 'clear' to start over")

        relations = []

        while True:
            if relations:
                print(f"\nCurrent relations:")
                for r in relations:
                    head = next(e for e in entities if e.id == r.head)
                    tail = next(e for e in entities if e.id == r.tail)
                    print(f"  ({head.text}, {r.relation}, {tail.text})")

            rel_input = input("\nEnter relation as 'head_id relation_id tail_id' (or 'done'): ").strip()

            if rel_input.lower() == 'done':
                break
            if rel_input.lower() == 'clear':
                relations = []
                continue

            parts = rel_input.split()
            if len(parts) != 3:
                print("  Format: head_id relation_id tail_id (e.g., '0 0 1')")
                continue

            try:
                head_id = int(parts[0])
                rel_id = int(parts[1])
                tail_id = int(parts[2])

                # Validate
                if head_id >= len(entities) or tail_id >= len(entities):
                    print(f"  Invalid entity ID. Max is {len(entities)-1}")
                    continue
                if rel_id >= len(RELATION_TYPES):
                    print(f"  Invalid relation ID. Max is {len(RELATION_TYPES)-1}")
                    continue

                relation = Relation(
                    head=head_id,
                    relation=RELATION_TYPES[rel_id],
                    tail=tail_id
                )
                relations.append(relation)
                head = next(e for e in entities if e.id == head_id)
                tail = next(e for e in entities if e.id == tail_id)
                print(f"  Added: ({head.text}, {RELATION_TYPES[rel_id]}, {tail.text})")

            except ValueError:
                print("  Invalid input. Use numbers for IDs.")
                continue

        return relations

    def annotate_sentence(self, sentence_id: str, text: str) -> Optional[Annotation]:
        """Annotate a single sentence."""
        print("\n" + "="*70)
        print(f"ANNOTATING: {sentence_id}")
        print("="*70)

        # Check if already annotated
        if any(a.sentence_id == sentence_id for a in self.annotations):
            skip = input("Already annotated. Skip? (y/n): ").strip().lower()
            if skip == 'y':
                return None

        # Annotate entities
        entities = self.annotate_entities(text)

        if not entities:
            print("\nNo entities annotated.")
            save = input("Save empty annotation? (y/n): ").strip().lower()
            if save != 'y':
                return None

        # Annotate relations
        relations = self.annotate_relations(text, entities)

        # Optional notes
        notes = input("\nAdd notes (optional): ").strip()

        # Create annotation
        ann = Annotation(
            sentence_id=sentence_id,
            text=text,
            entities=entities,
            relations=relations,
            annotator=self.annotator,
            notes=notes
        )

        return ann

    def run(self, start_idx: int = 0, max_count: int = 100):
        """Run interactive annotation session."""
        # Load sentences
        df = pd.read_csv(self.input_file)

        print(f"\n{'='*70}")
        print("HOST-PATHOGEN ANNOTATION TOOL")
        print(f"{'='*70}")
        print(f"Input: {self.input_file}")
        print(f"Output: {self.output_file}")
        print(f"Total sentences: {len(df)}")
        print(f"Already annotated: {len(self.annotations)}")
        print(f"Starting at index: {start_idx}")

        count = 0
        for idx, row in df.iloc[start_idx:].iterrows():
            if count >= max_count:
                break

            sentence_id = f"hp_{idx:05d}"
            text = row['passage']

            # Skip if already annotated
            if any(a.sentence_id == sentence_id for a in self.annotations):
                continue

            ann = self.annotate_sentence(sentence_id, text)

            if ann:
                self.annotations.append(ann)
                self._save_annotation(ann)
                print(f"\n>>> Saved annotation for {sentence_id}")
                count += 1

            # Continue prompt
            cont = input("\nContinue? (y/n/skip): ").strip().lower()
            if cont == 'n':
                break
            elif cont == 'skip':
                continue

        print(f"\n{'='*70}")
        print(f"SESSION COMPLETE")
        print(f"Annotated: {count} sentences")
        print(f"Total annotations: {len(self.annotations)}")
        print(f"Output: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Annotate host-pathogen sentences")
    parser.add_argument('--input', '-i', type=Path, required=True,
                       help='Input CSV file with sentences')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='Output JSONL file for annotations')
    parser.add_argument('--annotator', '-a', type=str, default='anonymous',
                       help='Annotator name')
    parser.add_argument('--start', '-s', type=int, default=0,
                       help='Starting index')
    parser.add_argument('--count', '-c', type=int, default=100,
                       help='Maximum sentences to annotate')

    args = parser.parse_args()

    tool = AnnotationTool(args.input, args.output, args.annotator)
    tool.run(start_idx=args.start, max_count=args.count)


if __name__ == "__main__":
    main()
