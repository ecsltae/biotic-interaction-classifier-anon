#!/usr/bin/env python3
"""
curate_ui.py — Beautiful terminal curation interface for biotic interaction sentences.

Usage:
  python classifier/tools/curate_ui.py import  <csv> --source <tag>
  python classifier/tools/curate_ui.py curate  [--source <tag>] [--n 20]
  python classifier/tools/curate_ui.py review  [--source <tag>]
  python classifier/tools/curate_ui.py stats
  python classifier/tools/curate_ui.py export  <output.csv> [--min-confidence 0.7]

Colors:
  • Green  = positive (biotic interaction)
  • Red    = negative (not an interaction)
  • Yellow = uncertain / needs human review
  • Cyan   = species names, interaction terms
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure classifier root on path
CLASSIFIER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.rule import Rule
from rich.align import Align
from rich.columns import Columns
from rich.live import Live
from rich import box

import tools.curation_db as db

console = Console()

# ── Palette ────────────────────────────────────────────────────────────────
C_POS       = "bold green"
C_NEG       = "bold red"
C_UNCERTAIN = "bold yellow"
C_SKIP      = "dim"
C_SPECIES   = "bold cyan"
C_TERM      = "italic magenta"
C_HEADER    = "bold white on dark_blue"
C_MUTED     = "grey62"
C_ACCENT    = "bold bright_cyan"

STATUS_ICON = {
    "approved":  "[green]✓[/green]",
    "uncertain": "[yellow]?[/yellow]",
    "skip":      "[dim]–[/dim]",
    "pending":   "[cyan]○[/cyan]",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _highlight_text(text: str, species: list[str] | None = None) -> Text:
    """Return a Rich Text object with species names highlighted."""
    t = Text(text)
    if species:
        for sp in species:
            if sp and len(sp) > 3:
                t.highlight_words([sp], style=C_SPECIES)
    # Highlight common interaction verbs
    _VERBS = [
        "parasitize", "parasitises", "infects", "infected by", "transmitted by",
        "vector of", "feeds on", "preys on", "pollinate", "disperses", "host of",
        "symbiont", "herbivore", "predator", "parasite", "pathogen",
    ]
    for v in _VERBS:
        t.highlight_words([v], style=C_TERM)
    return t


def _label_badge(label: int | None) -> str:
    if label == 1:
        return f"[{C_POS}]● POS[/{C_POS}]"
    elif label == 0:
        return f"[{C_NEG}]● NEG[/{C_NEG}]"
    return f"[{C_MUTED}]  ??? [/{C_MUTED}]"


def _confidence_bar(conf: float | None) -> str:
    if conf is None:
        return f"[{C_MUTED}]──────[/{C_MUTED}]"
    filled = round(conf * 10)
    bar = "█" * filled + "░" * (10 - filled)
    colour = C_POS if conf >= 0.7 else (C_UNCERTAIN if conf >= 0.4 else C_NEG)
    return f"[{colour}]{bar}[/{colour}] [{C_MUTED}]{conf:.2f}[/{C_MUTED}]"


# ── Banner ──────────────────────────────────────────────────────────────────

def _banner():
    console.print()
    console.print(Rule(
        "[bold bright_cyan]  MetaP Curation Interface  [/bold bright_cyan]",
        style="bright_cyan",
    ))
    console.print(
        Align.center(
            f"[{C_MUTED}]Biotic Interaction Training Data Validator[/{C_MUTED}]"
        )
    )
    console.print()


# ── IMPORT ──────────────────────────────────────────────────────────────────

def cmd_import(args):
    _banner()
    path = Path(args.csv)
    if not path.exists():
        console.print(f"[bold red]✗ File not found:[/bold red] {path}")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Source CSV:[/bold]  {path}\n"
        f"[bold]Source tag:[/bold]  [{C_ACCENT}]{args.source}[/{C_ACCENT}]",
        title="[bold]Import[/bold]",
        border_style="bright_cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading sentences into curation queue…", total=None)
        result = db.import_csv(path, args.source)
        progress.update(task, completed=True)

    console.print()
    console.print(Panel(
        f"  [green]✓ Imported:[/green]           [bold]{result['imported']:>6,}[/bold] new sentences\n"
        f"  [{C_MUTED}]⟳ Skipped (duplicate):[/{C_MUTED}]  [{C_MUTED}]{result['skipped_duplicates']:>6,}[/{C_MUTED}]\n"
        f"  [{C_ACCENT}]◉ Queue total (pending):[/{C_ACCENT}] [bold]{result['pending_total']:>6,}[/bold]",
        title=f"[green]Import complete — {args.source}[/green]",
        border_style="green",
    ))
    console.print()


# ── STATS ────────────────────────────────────────────────────────────────────

def cmd_stats(args):
    _banner()
    stats = db.get_stats()
    by_source = stats["by_source"]

    if not by_source:
        console.print(f"[{C_MUTED}]No data in curation queue yet. Use [bold]import[/bold] first.[/{C_MUTED}]")
        return

    table = Table(
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style=C_HEADER,
        show_lines=True,
        padding=(0, 1),
        title="[bold bright_cyan]Curation Queue Statistics[/bold bright_cyan]",
    )
    table.add_column("Source",    style="bold white", min_width=22)
    table.add_column("Total",     justify="right",  style="bold")
    table.add_column("Pending",   justify="right",  style="cyan")
    table.add_column("Approved",  justify="right",  style="green")
    table.add_column("Uncertain", justify="right",  style="yellow")
    table.add_column("Skip",      justify="right",  style=C_MUTED)
    table.add_column("Pos rate",  justify="right")
    table.add_column("Progress",  min_width=18)

    grand = {"total": 0, "pending": 0, "approved": 0, "uncertain": 0, "skip": 0}

    for source, s in sorted(by_source.items()):
        done = (s["approved"] or 0) + (s["uncertain"] or 0) + (s["skip"] or 0)
        pct = done / s["total"] if s["total"] else 0
        filled = round(pct * 16)
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (16 - filled)}[/dim] [{C_MUTED}]{pct:.0%}[/{C_MUTED}]"

        pos_r = s["pos_rate"]
        pos_str = f"{pos_r:.1%}" if pos_r is not None else "—"
        pos_col = C_POS if pos_r and pos_r > 0.6 else (C_NEG if pos_r and pos_r < 0.2 else "white")

        table.add_row(
            source,
            str(s["total"]),
            str(s["pending"] or 0),
            str(s["approved"] or 0),
            str(s["uncertain"] or 0),
            str(s["skip"] or 0),
            f"[{pos_col}]{pos_str}[/{pos_col}]",
            bar,
        )
        for k in grand:
            grand[k] += s.get(k) or 0

    table.add_section()
    done_grand = grand["approved"] + grand["uncertain"] + grand["skip"]
    pct_grand = done_grand / grand["total"] if grand["total"] else 0
    filled = round(pct_grand * 16)
    bar_grand = f"[green]{'█' * filled}[/green][dim]{'░' * (16 - filled)}[/dim] [{C_MUTED}]{pct_grand:.0%}[/{C_MUTED}]"
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{grand['total']}[/bold]",
        f"[cyan]{grand['pending']}[/cyan]",
        f"[green]{grand['approved']}[/green]",
        f"[yellow]{grand['uncertain']}[/yellow]",
        f"[dim]{grand['skip']}[/dim]",
        "—",
        bar_grand,
        style="bold",
    )

    console.print(table)

    if stats["claude_human_agreement"] is not None:
        agr = stats["claude_human_agreement"]
        col = C_POS if agr >= 0.85 else (C_UNCERTAIN if agr >= 0.7 else C_NEG)
        console.print(f"\n  Claude ↔ Human agreement: [{col}]{agr:.1%}[/{col}]\n")
    console.print()


# ── CURATE ──────────────────────────────────────────────────────────────────

_CURATE_HELP = (
    f"  [bold green]y[/bold green] / [bold green]1[/bold green]  → positive (biotic interaction)\n"
    f"  [bold red]n[/bold red] / [bold red]0[/bold red]  → negative\n"
    f"  [bold yellow]?[/bold yellow]        → uncertain (flag for review)\n"
    f"  [dim]s[/dim]        → skip (don't add to training)\n"
    f"  [dim]q[/dim]        → quit and save progress"
)


def _curate_item(item: dict, idx: int, total: int) -> tuple[int, float, str] | None:
    """Interactive curation for a single item. Returns (label, confidence, reasoning) or None to quit."""
    console.print()
    console.print(Rule(
        f"[{C_MUTED}]Item {idx}/{total}  ·  ID {item['id']}  ·  [{C_ACCENT}]{item['source']}[/{C_ACCENT}][/{C_MUTED}]",
        style="bright_cyan",
    ))

    # Sentence panel
    species = [item.get("source_species", ""), item.get("target_species", "")]
    rich_text = _highlight_text(item["text"], species)
    console.print(Panel(
        rich_text,
        border_style="white",
        padding=(1, 2),
    ))

    # Metadata row
    meta_parts = []
    orig = item.get("orig_label", -1)
    if orig == 1:
        meta_parts.append(f"Original label: [{C_POS}]POS[/{C_POS}]")
    elif orig == 0:
        meta_parts.append(f"Original label: [{C_NEG}]NEG[/{C_NEG}]")
    else:
        meta_parts.append(f"Original label: [{C_MUTED}]unknown[/{C_MUTED}]")

    if item.get("source_species"):
        meta_parts.append(f"Species 1: [{C_SPECIES}]{item['source_species']}[/{C_SPECIES}]")
    if item.get("target_species"):
        meta_parts.append(f"Species 2: [{C_SPECIES}]{item['target_species']}[/{C_SPECIES}]")
    if item.get("interaction_type"):
        meta_parts.append(f"Type: [{C_TERM}]{item['interaction_type']}[/{C_TERM}]")

    console.print(Columns(meta_parts, equal=False, expand=False), style=C_MUTED)
    console.print()
    console.print(_CURATE_HELP)
    console.print()

    while True:
        raw = Prompt.ask(
            f"  [{C_ACCENT}]Decision[/{C_ACCENT}]",
            default="?",
        ).strip().lower()

        if raw in ("q", "quit", "exit"):
            return None

        if raw in ("y", "1", "yes"):
            label, confidence = 1, 0.9
            reasoning = Prompt.ask(
                f"  [{C_MUTED}]Reasoning (Enter to skip)[/{C_MUTED}]",
                default="Positive: biotic interaction confirmed.",
            )
            break

        elif raw in ("n", "0", "no"):
            label, confidence = 0, 0.9
            reasoning = Prompt.ask(
                f"  [{C_MUTED}]Reasoning (Enter to skip)[/{C_MUTED}]",
                default="Negative: no direct biotic interaction.",
            )
            break

        elif raw == "?":
            label, confidence = -1, 0.5  # Will map to uncertain (label stored as orig, status=uncertain)
            reasoning = Prompt.ask(
                f"  [{C_UNCERTAIN}]Why uncertain?[/{C_UNCERTAIN}]",
                default="Uncertain — needs review.",
            )
            # For uncertain: keep orig_label but set confidence low
            label = item.get("orig_label", 0)
            if label == -1:
                label = 0
            confidence = 0.5
            break

        elif raw == "s":
            return (-1, 0.0, "skipped")

        else:
            console.print(f"  [{C_MUTED}]Use: y / n / ? / s / q[/{C_MUTED}]")

    return (label, confidence, reasoning)


def cmd_curate(args):
    _banner()
    n = args.n
    source = args.source

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
        t = p.add_task("Loading curation queue…", total=None)
        items = db.get_pending(source=source, n=n)
        p.update(t, completed=True)

    if not items:
        filter_msg = f" for source [bold]{source}[/bold]" if source else ""
        console.print(Panel(
            f"[green]✓ Nothing pending{filter_msg}![/green]  All sentences have been curated.",
            border_style="green",
        ))
        return

    console.print(Panel(
        f"  Source filter: [{C_ACCENT}]{source or 'all'}[/{C_ACCENT}]\n"
        f"  Items to curate: [bold]{len(items)}[/bold]\n\n"
        + _CURATE_HELP,
        title="[bold]Curation session[/bold]",
        border_style="bright_cyan",
    ))

    approved = uncertain = skipped = 0

    with Progress(
        SpinnerColumn(),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        ptask = progress.add_task("", total=len(items))

        for i, item in enumerate(items, 1):
            progress.update(
                ptask,
                advance=0,
                description=(
                    f"  [green]✓ {approved}[/green]  "
                    f"[yellow]? {uncertain}[/yellow]  "
                    f"[dim]– {skipped}[/dim]  "
                    f"  item {i}/{len(items)}"
                ),
            )

            result = _curate_item(item, i, len(items))

            if result is None:
                # User quit
                console.print(f"\n  [{C_MUTED}]Session interrupted — progress saved.[/{C_MUTED}]")
                break

            label, confidence, reasoning = result

            if label == -1:
                db.submit_decision(item["id"], 0, 0.0, reasoning, author="human")
                skipped += 1
            elif confidence < 0.7:
                db.submit_decision(item["id"], label, confidence, reasoning, author="human")
                uncertain += 1
            else:
                db.submit_decision(item["id"], label, confidence, reasoning, author="human")
                approved += 1

            progress.advance(ptask)

    # Summary
    console.print()
    console.print(Panel(
        f"  [green]✓ Approved:[/green]   [bold]{approved}[/bold]\n"
        f"  [yellow]? Uncertain:[/yellow]  [bold]{uncertain}[/bold]  (use [bold]review[/bold] to re-examine)\n"
        f"  [dim]– Skipped:[/dim]    [bold]{skipped}[/bold]",
        title="[bold green]Session complete[/bold green]",
        border_style="green",
    ))
    console.print()


# ── REVIEW ───────────────────────────────────────────────────────────────────

def cmd_review(args):
    _banner()
    items = db.list_decisions(
        uncertain_only=True,
        source=args.source if hasattr(args, "source") else None,
    )

    if not items:
        console.print(Panel(
            "[green]✓ No uncertain items to review![/green]",
            border_style="green",
        ))
        return

    console.print(Panel(
        f"  [yellow]{len(items)}[/yellow] items flagged as uncertain.\n"
        f"  Review each and confirm or override Claude's decision.\n\n"
        + _CURATE_HELP,
        title="[bold yellow]Human Review — Uncertain Items[/bold yellow]",
        border_style="yellow",
    ))

    resolved = 0
    for i, item in enumerate(items, 1):
        console.print()
        console.print(Rule(
            f"[{C_MUTED}]Review {i}/{len(items)}  ·  ID {item['id']}  ·  "
            f"[{C_UNCERTAIN}]uncertain[/{C_UNCERTAIN}][/{C_MUTED}]",
            style="yellow",
        ))

        species = [item.get("source_species", ""), item.get("target_species", "")]
        console.print(Panel(
            _highlight_text(item["text"], species),
            border_style="yellow",
            padding=(1, 2),
        ))

        # Show prior decision
        if item.get("reasoning"):
            console.print(f"  [{C_MUTED}]Prior reasoning: {item['reasoning']}[/{C_MUTED}]")
        orig_label_str = _label_badge(item.get("orig_label"))
        cur_label_str = _label_badge(item.get("label"))
        console.print(
            f"  Orig label: {orig_label_str}   "
            f"Current: {cur_label_str}   "
            f"Confidence: {_confidence_bar(item.get('confidence'))}"
        )
        console.print()
        console.print(_CURATE_HELP)
        console.print()

        while True:
            raw = Prompt.ask(f"  [{C_ACCENT}]Override[/{C_ACCENT}]", default="?").strip().lower()
            if raw in ("q", "quit", "exit"):
                console.print(f"\n  [{C_MUTED}]Review interrupted — progress saved.[/{C_MUTED}]")
                return
            if raw in ("y", "1", "yes"):
                label, confidence = 1, 0.95
                reasoning = Prompt.ask(f"  [{C_MUTED}]Reasoning[/{C_MUTED}]", default="Human confirmed: positive.")
                break
            elif raw in ("n", "0", "no"):
                label, confidence = 0, 0.95
                reasoning = Prompt.ask(f"  [{C_MUTED}]Reasoning[/{C_MUTED}]", default="Human confirmed: negative.")
                break
            elif raw == "?":
                console.print(f"  [{C_MUTED}]Still uncertain — skipping.[/{C_MUTED}]")
                label, confidence, reasoning = (item.get("label") or 0), 0.5, item.get("reasoning", "")
                break
            elif raw == "s":
                label, confidence, reasoning = -1, 0.0, "skipped"
                break
            else:
                console.print(f"  [{C_MUTED}]Use: y / n / ? / s / q[/{C_MUTED}]")

        if label != -1:
            db.submit_decision(item["id"], label, confidence, reasoning, author="human")
            resolved += 1

    console.print()
    console.print(Panel(
        f"  [green]✓ Resolved:[/green]  [bold]{resolved}[/bold] / {len(items)}",
        title="[bold]Review complete[/bold]",
        border_style="green",
    ))
    console.print()


# ── EXPORT ───────────────────────────────────────────────────────────────────

def cmd_export(args):
    _banner()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    min_conf = args.min_confidence

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
        t = p.add_task("Exporting approved decisions…", total=None)
        result = db.export_approved(out_path, min_confidence=min_conf)
        p.update(t, completed=True)

    if result["exported"] == 0:
        console.print(Panel(
            f"[yellow]No approved decisions with confidence ≥ {min_conf:.2f}.[/yellow]\n"
            f"Run [bold]curate[/bold] or [bold]review[/bold] to build up decisions first.",
            border_style="yellow",
        ))
        return

    pos_pct = result["pos"] / result["exported"] if result["exported"] else 0
    console.print(Panel(
        f"  [green]✓ Exported:[/green]  [bold]{result['exported']:,}[/bold] sentences\n"
        f"  [{C_POS}]● Positive:[/{C_POS}]  [bold]{result['pos']:,}[/bold]  "
        f"[{C_NEG}]● Negative:[/{C_NEG}]  [bold]{result['neg']:,}[/bold]  "
        f"[{C_MUTED}]({pos_pct:.1%} pos)[/{C_MUTED}]\n\n"
        f"  Path:  [{C_ACCENT}]{result['path']}[/{C_ACCENT}]\n\n"
        f"  [{C_MUTED}]Ready to use as --extra-sources in build_v*_dataset.py[/{C_MUTED}]",
        title="[bold green]Export complete[/bold green]",
        border_style="green",
    ))
    console.print()

    # Suggest next step
    console.print(
        f"  [{C_MUTED}]Next step:[/{C_MUTED}]\n"
        f"  [bold]python classifier/scripts/build_v12_dataset.py \\\n"
        f"    --extra-sources {result['path']} \\\n"
        f"    --output classifier/data/training/training_data_v13b.csv[/bold]"
    )
    console.print()


# ── LIST ─────────────────────────────────────────────────────────────────────

def cmd_list(args):
    _banner()
    decisions = db.list_decisions(
        status=getattr(args, "status", None),
        source=getattr(args, "source", None),
    )

    if not decisions:
        console.print(f"[{C_MUTED}]No decisions found.[/{C_MUTED}]")
        return

    table = Table(
        box=box.SIMPLE_HEAD,
        border_style="bright_cyan",
        header_style=C_HEADER,
        show_lines=False,
        padding=(0, 1),
        title=f"[bold bright_cyan]Curation Decisions ({len(decisions)})[/bold bright_cyan]",
    )
    table.add_column("ID",     justify="right", style=C_MUTED, width=6)
    table.add_column("St",     justify="center", width=3)
    table.add_column("Lbl",    justify="center", width=5)
    table.add_column("Conf",   justify="right",  width=5)
    table.add_column("Auth",   width=7)
    table.add_column("Source", width=18)
    table.add_column("Text",   no_wrap=False, max_width=70)

    for d in decisions[:200]:  # cap display
        status_icon = STATUS_ICON.get(d.get("status", ""), "·")
        label_str = _label_badge(d.get("label"))
        conf = d.get("confidence")
        conf_str = f"{conf:.2f}" if conf else "—"
        author = d.get("author") or "—"
        author_col = C_ACCENT if author == "human" else C_MUTED
        text_preview = (d["text"][:80] + "…") if len(d["text"]) > 80 else d["text"]

        table.add_row(
            str(d["id"]),
            status_icon,
            label_str,
            conf_str,
            f"[{author_col}]{author}[/{author_col}]",
            d.get("source") or "—",
            text_preview,
        )

    console.print(table)
    if len(decisions) > 200:
        console.print(f"  [{C_MUTED}]… {len(decisions) - 200} more rows not shown.[/{C_MUTED}]")
    console.print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="curate_ui",
        description="MetaP Curation Interface — validate training sentences interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # import
    p_import = sub.add_parser("import", help="Load a CSV into the curation queue")
    p_import.add_argument("csv", help="Path to source CSV")
    p_import.add_argument("--source", required=True, help="Source tag (e.g. sibils_mongodb)")

    # curate
    p_curate = sub.add_parser("curate", help="Interactively curate pending sentences")
    p_curate.add_argument("--source", default=None, help="Filter by source tag")
    p_curate.add_argument("--n", type=int, default=20, help="Number of items per session")

    # review
    p_review = sub.add_parser("review", help="Human review of uncertain items")
    p_review.add_argument("--source", default=None, help="Filter by source tag")

    # stats
    sub.add_parser("stats", help="Show queue statistics")

    # export
    p_export = sub.add_parser("export", help="Export approved decisions to CSV")
    p_export.add_argument("output", help="Output CSV path")
    p_export.add_argument("--min-confidence", type=float, default=0.7, dest="min_confidence")

    # list
    p_list = sub.add_parser("list", help="List decisions in a table")
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--source", default=None)

    args = parser.parse_args()

    dispatch = {
        "import":  cmd_import,
        "curate":  cmd_curate,
        "review":  cmd_review,
        "stats":   cmd_stats,
        "export":  cmd_export,
        "list":    cmd_list,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
