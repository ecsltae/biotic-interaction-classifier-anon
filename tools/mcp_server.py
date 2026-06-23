#!/usr/bin/env python3
"""
MCP Server for Biotic Interaction Classification.

Exposes tools for validating and classifying biotic interaction sentences,
plus a full curation workflow (import → queue → decide → export).

Run directly: python classifier/tools/mcp_server.py
Or via .mcp.json configuration in Claude Code.
"""
import sys
import json
import logging
from pathlib import Path

# Add classifier root to path so we can import validator module
CLASSIFIER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from validator.interaction_validator import (
    validate_interaction_sentence,
    batch_validate_sentences,
)
from tools.curation_db import (
    import_csv,
    get_pending,
    submit_decision,
    get_stats,
    list_decisions,
    export_approved,
)

logger = logging.getLogger(__name__)
server = Server("metap-interaction-validator")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools():
    return [
        # ── Existing validation tools ──────────────────────────────────────
        Tool(
            name="validate_interaction",
            description=(
                "Validate whether a sentence describes a biotic interaction between "
                "organisms. Returns confidence score and reasoning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sentence": {
                        "type": "string",
                        "description": "The sentence to validate for biotic interaction content",
                    },
                    "use_llm": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use LLM validation if ANTHROPIC_API_KEY is set, otherwise falls back to heuristics",
                    },
                },
                "required": ["sentence"],
            },
        ),
        Tool(
            name="batch_validate",
            description=(
                "Validate multiple sentences for biotic interactions. "
                "Returns only those above the confidence threshold."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sentences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sentences to validate",
                    },
                    "min_confidence": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum confidence threshold (0.0-1.0)",
                    },
                },
                "required": ["sentences"],
            },
        ),
        Tool(
            name="classify_text",
            description=(
                "Classify text using the trained ML ensemble API (BiomedBERT+RoBERTa). "
                "Requires the FastAPI to be running on port 8001."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to classify as biotic interaction or not",
                    }
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="get_training_stats",
            description="Get statistics about the current training dataset: row count, label distribution, latest version.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Curation workflow tools ────────────────────────────────────────
        Tool(
            name="import_for_curation",
            description=(
                "Load a source CSV into the curation queue for LLM/human validation. "
                "Deduplicates by sentence text. Call before get_curation_queue."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_csv": {
                        "type": "string",
                        "description": "Absolute or project-relative path to the CSV file to import",
                    },
                    "source_tag": {
                        "type": "string",
                        "description": "Short label for this source (e.g. 'sibils_mongodb', 'sibils_diverse')",
                    },
                },
                "required": ["source_csv", "source_tag"],
            },
        ),
        Tool(
            name="get_curation_queue",
            description=(
                "Fetch the next N sentences pending curation, sorted by uncertainty "
                "(disagreement between original label and heuristic score first). "
                "Returns id, text, orig_label, source_species, target_species for each item."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Filter by source tag (optional). Omit to get items from all sources.",
                    },
                    "n": {
                        "type": "integer",
                        "default": 20,
                        "description": "Number of items to return (default 20, max 100)",
                    },
                },
            },
        ),
        Tool(
            name="submit_curation_decision",
            description=(
                "Save a label + reasoning for a curation queue item. "
                "confidence >= 0.7 → 'approved'; 0.4–0.7 → 'uncertain' (flagged for human review); "
                "label -1 → 'skip'. Author should be 'claude' or 'human'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Queue item ID (from get_curation_queue)",
                    },
                    "label": {
                        "type": "integer",
                        "description": "1 = biotic interaction, 0 = not, -1 = skip",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0–1.0",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the decision",
                    },
                    "author": {
                        "type": "string",
                        "default": "claude",
                        "description": "'claude' or 'human'",
                    },
                },
                "required": ["id", "label", "confidence", "reasoning"],
            },
        ),
        Tool(
            name="get_curation_stats",
            description=(
                "Get per-source breakdown of the curation queue: "
                "pending/approved/uncertain/skip counts and positive rate. "
                "Also reports Claude vs human agreement when both have annotated the same items."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_curation_decisions",
            description=(
                "List saved curation decisions with optional filters. "
                "Use uncertain_only=true to get items flagged for human review."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: 'pending'|'approved'|'uncertain'|'skip'",
                    },
                    "author": {
                        "type": "string",
                        "description": "Filter by author: 'claude'|'human'",
                    },
                    "source": {
                        "type": "string",
                        "description": "Filter by source tag",
                    },
                    "uncertain_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, return only uncertain items regardless of other filters",
                    },
                },
            },
        ),
        Tool(
            name="export_curated_data",
            description=(
                "Export approved curation decisions to a training-ready CSV "
                "(columns: text, label, interaction_type, source_species, target_species, source). "
                "The output can be passed directly as --extra-sources to build_v*_dataset.py."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Path for the output CSV (e.g. 'classifier/data/training/curated_sibils.csv')",
                    },
                    "min_confidence": {
                        "type": "number",
                        "default": 0.7,
                        "description": "Only export decisions with confidence >= this value",
                    },
                    "author_filter": {
                        "type": "string",
                        "description": "Only export decisions from this author (optional)",
                    },
                },
                "required": ["output_path"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── validate_interaction ────────────────────────────────────────────────
    if name == "validate_interaction":
        result = validate_interaction_sentence(
            arguments["sentence"],
            use_llm=arguments.get("use_llm", True),
        )
        return [TextContent(type="text", text=json.dumps(result.to_dict(), indent=2))]

    # ── batch_validate ──────────────────────────────────────────────────────
    elif name == "batch_validate":
        results = batch_validate_sentences(
            arguments["sentences"],
            min_confidence=arguments.get("min_confidence", 0.5),
        )
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    # ── classify_text ───────────────────────────────────────────────────────
    elif name == "classify_text":
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:8001/predict",
                    json={"text": arguments["text"]},
                    timeout=30,
                )
                return [TextContent(type="text", text=resp.text)]
        except httpx.ConnectError:
            return [TextContent(type="text", text=json.dumps({
                "error": "API not running",
                "hint": "Start with: bash classifier/start_api.sh",
            }))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    # ── get_training_stats ──────────────────────────────────────────────────
    elif name == "get_training_stats":
        import pandas as pd
        data_dir = CLASSIFIER_ROOT / "data" / "training"
        csvs = sorted(data_dir.glob("training_data_v*.csv"))
        if csvs:
            latest = csvs[-1]
            df = pd.read_csv(latest)
            stats = {
                "file": latest.name,
                "total_rows": len(df),
                "columns": list(df.columns),
            }
            if "label" in df.columns:
                stats["label_distribution"] = df["label"].value_counts().to_dict()
        else:
            stats = {"error": "No training data found in " + str(data_dir)}
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    # ── import_for_curation ─────────────────────────────────────────────────
    elif name == "import_for_curation":
        csv_path = arguments["source_csv"]
        # Resolve relative paths from project root
        p = Path(csv_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        result = import_csv(p, arguments["source_tag"])
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── get_curation_queue ──────────────────────────────────────────────────
    elif name == "get_curation_queue":
        n = min(int(arguments.get("n", 20)), 100)
        items = get_pending(source=arguments.get("source"), n=n)
        return [TextContent(type="text", text=json.dumps(items, indent=2))]

    # ── submit_curation_decision ────────────────────────────────────────────
    elif name == "submit_curation_decision":
        result = submit_decision(
            item_id=int(arguments["id"]),
            label=int(arguments["label"]),
            confidence=float(arguments["confidence"]),
            reasoning=arguments["reasoning"],
            author=arguments.get("author", "claude"),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── get_curation_stats ──────────────────────────────────────────────────
    elif name == "get_curation_stats":
        stats = get_stats()
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    # ── list_curation_decisions ─────────────────────────────────────────────
    elif name == "list_curation_decisions":
        decisions = list_decisions(
            status=arguments.get("status"),
            author=arguments.get("author"),
            source=arguments.get("source"),
            uncertain_only=arguments.get("uncertain_only", False),
        )
        return [TextContent(type="text", text=json.dumps(decisions, indent=2))]

    # ── export_curated_data ─────────────────────────────────────────────────
    elif name == "export_curated_data":
        out = arguments["output_path"]
        p = Path(out)
        if not p.is_absolute():
            p = Path.cwd() / p
        result = export_approved(
            output_path=p,
            min_confidence=float(arguments.get("min_confidence", 0.7)),
            author_filter=arguments.get("author_filter"),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
