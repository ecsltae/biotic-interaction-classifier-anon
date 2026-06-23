#!/usr/bin/env python3
"""
curate_web.py — Gradio web interface for biotic interaction curation.

Usage:
  source MPvenv/bin/activate
  python classifier/tools/curate_web.py [--port 7860] [--source sibils_mongodb]

Tabs:
  Dashboard  — per-source progress + agreement stats
  Review     — uncertain items flagged for human confirmation
  Curate     — pending items (unreviewed)
  Export     — export approved decisions to training CSV
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CLASSIFIER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import curation_db as db

# ---------------------------------------------------------------------------
# Palette (CSS variables injected via custom_css)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── global ─────────────────────────────────────── */
.gradio-container { max-width: 960px !important; margin: auto; }

/* ── sentence card ──────────────────────────────── */
.sentence-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 1.05rem;
    line-height: 1.75;
    color: #cdd6f4;
    min-height: 80px;
}
.sentence-card .sp { color: #89dceb; font-weight: 600; }   /* species  */
.sentence-card .vb { color: #f9e2af; font-style: italic; } /* verb     */

/* ── meta badges ────────────────────────────────── */
.meta-row {
    display: flex; gap: 12px; flex-wrap: wrap;
    margin-top: 10px; font-size: 0.85rem;
}
.badge {
    border-radius: 6px; padding: 2px 10px; font-weight: 600;
}
.badge-pos   { background: #a6e3a1; color: #1e1e2e; }
.badge-neg   { background: #f38ba8; color: #1e1e2e; }
.badge-unk   { background: #45475a; color: #cdd6f4; }
.badge-type  { background: #cba6f7; color: #1e1e2e; }
.badge-sp    { background: #89dceb; color: #1e1e2e; }

/* ── decision buttons ───────────────────────────── */
.btn-pos  { background: #40a02b !important; color: #fff !important;
            border: none !important; font-size: 1.1rem !important; }
.btn-neg  { background: #d20f39 !important; color: #fff !important;
            border: none !important; font-size: 1.1rem !important; }
.btn-unc  { background: #df8e1d !important; color: #fff !important;
            border: none !important; font-size: 1.1rem !important; }
.btn-skip { background: #45475a !important; color: #cdd6f4 !important;
            border: none !important; }

/* ── progress bar (stats) ───────────────────────── */
.prog-wrap { background: #313244; border-radius: 4px; height: 10px; width: 100%; }
.prog-fill { background: #a6e3a1; border-radius: 4px; height: 10px; }

/* ── counter pill ───────────────────────────────── */
.counter {
    text-align: center; font-size: 1rem; color: #a6adc8;
    margin-bottom: 4px;
}
.counter span { color: #cba6f7; font-weight: 700; font-size: 1.3rem; }

/* ── reasoning box ──────────────────────────────── */
.reasoning-hint { color: #6c7086; font-size: 0.82rem; margin-top: 4px; }
"""

# Interaction verbs to highlight
_VERBS = re.compile(
    r'\b(parasitiz\w+|infect\w*|transmitted? by|vector of|feeds? on|preys? on|'
    r'pollinat\w+|dispers\w+|host of|symbiont\w*|herbivore|predator|parasite|'
    r'pathogen|coloniz\w+|grazes? on|eats?|consumes?|lays? eggs|oviposit\w+)\b',
    re.IGNORECASE,
)


def _render_sentence(item: dict) -> str:
    """Return HTML for the sentence panel with highlighted species + verbs."""
    text = item.get("text", "")
    sp1 = (item.get("source_species") or "").strip()
    sp2 = (item.get("target_species") or "").strip()

    # Escape HTML
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    safe = _esc(text)

    # Highlight species (longest first to avoid partial matches)
    for sp in sorted([sp1, sp2], key=len, reverse=True):
        if sp and len(sp) > 3:
            safe = re.sub(
                r'(?i)(' + re.escape(_esc(sp)) + r')',
                r'<span class="sp">\1</span>',
                safe,
            )
    # Highlight verbs
    safe = _VERBS.sub(lambda m: f'<span class="vb">{_esc(m.group())}</span>', safe)

    # Build meta badges
    orig = item.get("orig_label", -1)
    orig_badge = (
        '<span class="badge badge-pos">ORIG: POS</span>' if orig == 1 else
        '<span class="badge badge-neg">ORIG: NEG</span>' if orig == 0 else
        '<span class="badge badge-unk">ORIG: ?</span>'
    )
    badges = [orig_badge]
    if sp1:
        badges.append(f'<span class="badge badge-sp">{_esc(sp1)}</span>')
    if sp2:
        badges.append(f'<span class="badge badge-sp">{_esc(sp2)}</span>')
    itype = (item.get("interaction_type") or "").strip()
    if itype:
        badges.append(f'<span class="badge badge-type">{_esc(itype)}</span>')

    reasoning = item.get("reasoning") or ""
    reasoning_html = (
        f'<div class="reasoning-hint">⚠ Prior reasoning: {_esc(reasoning)}</div>'
        if reasoning else ""
    )

    source = item.get("source") or ""
    source_file = item.get("source_file") or ""
    item_id = item.get("id", "?")

    source_file_html = (
        f'&nbsp;·&nbsp; <span style="color:#89b4fa;font-weight:600">{_esc(source_file)}</span>'
        if source_file else ""
    )

    return f"""
<div>
  <div style="font-size:0.78rem;color:#6c7086;margin-bottom:8px;">
    ID {item_id} &nbsp;·&nbsp; {_esc(source)}{source_file_html}
  </div>
  <div class="sentence-card">{safe}</div>
  <div class="meta-row">{''.join(badges)}</div>
  {reasoning_html}
</div>
"""


def _render_empty(msg: str = "Nothing to review here.") -> str:
    return f"""
<div class="sentence-card" style="color:#6c7086;text-align:center;padding:40px;">
  ✓ {msg}
</div>
"""


def _counter_html(idx: int, total: int, label: str = "items") -> str:
    if total == 0:
        return f'<div class="counter">No {label}</div>'
    return f'<div class="counter"><span>{idx}</span> / {total} {label}</div>'


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _stats_html() -> str:
    stats = db.get_stats()
    by_source = stats["by_source"]
    if not by_source:
        return "<p style='color:#6c7086'>No data yet — import a source first.</p>"

    rows_html = ""
    for src, s in sorted(by_source.items()):
        total = s["total"] or 1
        done = (s["approved"] or 0) + (s["uncertain"] or 0) + (s["skip"] or 0)
        pct = done / total
        pos_r = s["pos_rate"]
        pos_str = f"{pos_r:.1%}" if pos_r is not None else "—"
        pos_col = "#a6e3a1" if (pos_r or 0) > 0.6 else ("#f38ba8" if (pos_r or 0) < 0.2 else "#f9e2af")

        rows_html += f"""
        <tr>
          <td style="font-weight:600;color:#cdd6f4">{src}</td>
          <td style="color:#a6adc8">{s['total']}</td>
          <td style="color:#89b4fa">{s['pending'] or 0}</td>
          <td style="color:#a6e3a1">{s['approved'] or 0}</td>
          <td style="color:#f9e2af">{s['uncertain'] or 0}</td>
          <td style="color:#6c7086">{s['skip'] or 0}</td>
          <td style="color:{pos_col}">{pos_str}</td>
          <td style="width:140px">
            <div class="prog-wrap">
              <div class="prog-fill" style="width:{pct:.0%}"></div>
            </div>
            <span style="font-size:0.75rem;color:#6c7086">{pct:.0%}</span>
          </td>
        </tr>"""

    agr = stats.get("claude_human_agreement")
    agr_html = ""
    if agr is not None:
        col = "#a6e3a1" if agr >= 0.85 else ("#f9e2af" if agr >= 0.7 else "#f38ba8")
        agr_html = f"""
        <p style="margin-top:16px;color:#a6adc8">
          Claude ↔ Human agreement: <strong style="color:{col}">{agr:.1%}</strong>
        </p>"""

    return f"""
<style>
  .stats-table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
  .stats-table th {{ text-align:left; padding:6px 12px; background:#313244 !important;
                     color:#e0d0ff !important; border-bottom:2px solid #45475a;
                     font-weight:700; letter-spacing:0.03em; }}
  .stats-table td {{ padding:8px 12px; border-bottom:1px solid #313244; }}
  .stats-table tr:hover td {{ background:#1e1e2e; }}
</style>
<table class="stats-table">
  <thead><tr>
    <th>Source</th><th>Total</th><th>Pending</th>
    <th>Approved</th><th>Uncertain</th><th>Skip</th>
    <th>Pos rate</th><th>Progress</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
{agr_html}
"""


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _load_uncertain(source: str | None) -> list[dict]:
    return db.list_decisions(uncertain_only=True, source=source or None)


def _load_pending(source: str | None, n: int) -> list[dict]:
    return db.get_pending(source=source or None, n=n)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_app(default_source: str = "") -> gr.Blocks:

    with gr.Blocks(title="MetaP Curation") as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🧬 MetaP Curation Interface\n"
            "Biotic interaction training data validator — "
            "review, curate, and export sentences for model training."
        )

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Dashboard
            # ================================================================
            with gr.Tab("📊 Dashboard"):
                stats_html = gr.HTML(value=_stats_html)
                refresh_btn = gr.Button("↻ Refresh", size="sm", variant="secondary")
                refresh_btn.click(fn=_stats_html, outputs=stats_html)

            # ================================================================
            # TAB 2 — Review (uncertain items + low-confidence approved)
            # ================================================================
            with gr.Tab("⚠️ Review — Uncertain"):

                with gr.Row():
                    review_source = gr.Dropdown(
                        label="Source filter",
                        choices=["", "sibils_mongodb", "sibils_diverse", "globi_sibils"],
                        value=default_source,
                        scale=2,
                    )
                    review_mode = gr.Radio(
                        label="Review mode",
                        choices=["Uncertain items", "Low-confidence approved"],
                        value="Uncertain items",
                        scale=2,
                    )
                    review_load_btn = gr.Button("Load items", variant="primary", scale=1)

                review_max_conf = gr.Slider(
                    label="Max confidence threshold (only for 'Low-confidence approved')",
                    minimum=0.5, maximum=0.95, step=0.05, value=0.85,
                    visible=False,
                )

                def _toggle_conf_slider(mode):
                    return gr.update(visible=(mode == "Low-confidence approved"))

                review_mode.change(fn=_toggle_conf_slider, inputs=review_mode, outputs=review_max_conf)

                review_counter  = gr.HTML(value=_counter_html(0, 0, "items"))
                review_sentence = gr.HTML(value=_render_empty("Click 'Load items' to start reviewing."))
                review_reasoning = gr.Textbox(
                    label="Override reasoning (optional)",
                    placeholder="Leave blank to keep prior reasoning…",
                    lines=2,
                )

                with gr.Row():
                    rv_pos  = gr.Button("✅  Positive", elem_classes="btn-pos",  variant="primary")
                    rv_neg  = gr.Button("❌  Negative", elem_classes="btn-neg",  variant="stop")
                    rv_unc  = gr.Button("❓  Still uncertain", elem_classes="btn-unc")
                    rv_skip = gr.Button("⏭  Skip", elem_classes="btn-skip", variant="secondary")

                review_status = gr.Markdown("")

                # State
                review_items  = gr.State([])
                review_idx    = gr.State(0)

                # ── Load ────────────────────────────────────────────────────
                def load_review(source, mode, max_conf):
                    if mode == "Low-confidence approved":
                        all_approved = db.list_decisions(
                            status="approved", source=source or None
                        )
                        items = [
                            i for i in all_approved
                            if (i.get("confidence") or 1.0) < max_conf
                            and i.get("label") == 1  # only re-check positives
                        ]
                        label = f"low-confidence approved (conf < {max_conf:.0%})"
                    else:
                        items = _load_uncertain(source)
                        label = "uncertain items"

                    if not items:
                        return (
                            [], 0,
                            _render_empty(f"No {label} for this source ✓"),
                            _counter_html(0, 0, label),
                            "",
                        )
                    return (
                        items, 0,
                        _render_sentence(items[0]),
                        _counter_html(1, len(items), label),
                        f"Loaded {len(items)} {label}.",
                    )

                review_load_btn.click(
                    fn=load_review,
                    inputs=[review_source, review_mode, review_max_conf],
                    outputs=[review_items, review_idx, review_sentence, review_counter, review_status],
                )

                # ── Decision handler ────────────────────────────────────────
                def _submit_review(items, idx, label, confidence, reasoning_override):
                    if not items or idx >= len(items):
                        return items, idx, _render_empty("All done ✓"), _counter_html(0, 0, "uncertain items"), ""

                    item = items[idx]
                    reason = reasoning_override.strip() or item.get("reasoning") or ""
                    db.submit_decision(item["id"], label, confidence, reason, author="human")

                    next_idx = idx + 1
                    if next_idx >= len(items):
                        return (
                            items, next_idx,
                            _render_empty("All items reviewed ✓"),
                            _counter_html(len(items), len(items), "uncertain items"),
                            f"✓ Done — reviewed all {len(items)} items.",
                        )
                    return (
                        items, next_idx,
                        _render_sentence(items[next_idx]),
                        _counter_html(next_idx + 1, len(items), "uncertain items"),
                        f"Saved item {item['id']}.",
                    )

                _review_outs = [review_items, review_idx, review_sentence, review_counter, review_status]

                rv_pos.click(
                    fn=lambda it, i, r: _submit_review(it, i, 1, 0.95, r),
                    inputs=[review_items, review_idx, review_reasoning],
                    outputs=_review_outs,
                )
                rv_neg.click(
                    fn=lambda it, i, r: _submit_review(it, i, 0, 0.95, r),
                    inputs=[review_items, review_idx, review_reasoning],
                    outputs=_review_outs,
                )
                rv_unc.click(
                    fn=lambda it, i, r: _submit_review(
                        it, i,
                        it[i].get("label") or 0 if it and i < len(it) else 0,
                        0.5, r,
                    ),
                    inputs=[review_items, review_idx, review_reasoning],
                    outputs=_review_outs,
                )
                rv_skip.click(
                    fn=lambda it, i, r: _submit_review(it, i, -1, 0.0, r),
                    inputs=[review_items, review_idx, review_reasoning],
                    outputs=_review_outs,
                )

            # ================================================================
            # TAB 3 — Curate (pending items)
            # ================================================================
            with gr.Tab("🔬 Curate — Pending"):

                with gr.Row():
                    _all_pending = db.get_pending(n=10000)
                    _curate_sources = [""] + sorted({r["source"] for r in _all_pending if r.get("source")})
                    curate_source = gr.Dropdown(
                        label="Source filter",
                        choices=_curate_sources,
                        value=default_source,
                        scale=2,
                    )
                    curate_n = gr.Slider(
                        label="Batch size",
                        minimum=5, maximum=100, step=5, value=20, scale=2,
                    )
                    curate_load_btn = gr.Button("Load batch", variant="primary", scale=1)

                curate_counter  = gr.HTML(value=_counter_html(0, 0, "pending items"))
                curate_sentence = gr.HTML(value=_render_empty("Click 'Load batch' to start curating."))
                curate_reasoning = gr.Textbox(
                    label="Reasoning (optional)",
                    placeholder="Why positive / negative? Leave blank for default.",
                    lines=2,
                )
                curate_confidence = gr.Slider(
                    label="Confidence",
                    minimum=0.0, maximum=1.0, step=0.05, value=0.9,
                )

                with gr.Row():
                    cu_pos  = gr.Button("✅  Positive", elem_classes="btn-pos",  variant="primary")
                    cu_neg  = gr.Button("❌  Negative", elem_classes="btn-neg",  variant="stop")
                    cu_unc  = gr.Button("❓  Uncertain", elem_classes="btn-unc")
                    cu_skip = gr.Button("⏭  Skip", elem_classes="btn-skip", variant="secondary")

                curate_status = gr.Markdown("")

                # State
                curate_items = gr.State([])
                curate_idx   = gr.State(0)

                # ── Load ────────────────────────────────────────────────────
                def load_curate(source, n):
                    items = _load_pending(source, int(n))
                    if not items:
                        return (
                            [], 0,
                            _render_empty("No pending items for this source ✓"),
                            _counter_html(0, 0, "pending items"),
                            "",
                        )
                    return (
                        items, 0,
                        _render_sentence(items[0]),
                        _counter_html(1, len(items), "pending items"),
                        "",
                    )

                curate_load_btn.click(
                    fn=load_curate,
                    inputs=[curate_source, curate_n],
                    outputs=[curate_items, curate_idx, curate_sentence, curate_counter, curate_status],
                )

                # ── Decision handler ────────────────────────────────────────
                def _submit_curate(items, idx, label, confidence, reasoning):
                    if not items or idx >= len(items):
                        return items, idx, _render_empty("Batch complete ✓"), _counter_html(0, 0, "pending items"), ""

                    item = items[idx]
                    reason = reasoning.strip() or (
                        "Positive: biotic interaction confirmed." if label == 1 else "Negative: no direct biotic interaction."
                    )
                    if label == -1:
                        db.submit_decision(item["id"], 0, 0.0, "skipped", author="human")
                    else:
                        db.submit_decision(item["id"], label, confidence, reason, author="human")

                    next_idx = idx + 1
                    if next_idx >= len(items):
                        return (
                            items, next_idx,
                            _render_empty("Batch complete — load another to continue ✓"),
                            _counter_html(len(items), len(items), "pending items"),
                            f"✓ Batch done — curated {len(items)} items.",
                        )
                    return (
                        items, next_idx,
                        _render_sentence(items[next_idx]),
                        _counter_html(next_idx + 1, len(items), "pending items"),
                        "",
                    )

                _curate_outs = [curate_items, curate_idx, curate_sentence, curate_counter, curate_status]

                cu_pos.click(
                    fn=lambda it, i, c, r: _submit_curate(it, i, 1, c, r),
                    inputs=[curate_items, curate_idx, curate_confidence, curate_reasoning],
                    outputs=_curate_outs,
                )
                cu_neg.click(
                    fn=lambda it, i, c, r: _submit_curate(it, i, 0, c, r),
                    inputs=[curate_items, curate_idx, curate_confidence, curate_reasoning],
                    outputs=_curate_outs,
                )
                cu_unc.click(
                    fn=lambda it, i, r: _submit_curate(it, i, it[i].get("orig_label") or 0 if it and i < len(it) else 0, 0.5, r),
                    inputs=[curate_items, curate_idx, curate_reasoning],
                    outputs=_curate_outs,
                )
                cu_skip.click(
                    fn=lambda it, i, r: _submit_curate(it, i, -1, 0.0, r),
                    inputs=[curate_items, curate_idx, curate_reasoning],
                    outputs=_curate_outs,
                )

                # Keyboard shortcuts hint
                gr.Markdown(
                    "_Tip: y = positive · n = negative · ? = uncertain · s = skip_",
                    elem_classes=["reasoning-hint"],
                )

            # ================================================================
            # TAB 4 — Export
            # ================================================================
            with gr.Tab("📤 Export"):

                gr.Markdown(
                    "Export approved decisions to a training-ready CSV that can be passed "
                    "directly to `build_v*_dataset.py --extra-sources`."
                )

                with gr.Row():
                    export_path = gr.Textbox(
                        label="Output path",
                        value="classifier/data/training/curated_export.csv",
                        scale=3,
                    )
                    export_min_conf = gr.Slider(
                        label="Min confidence",
                        minimum=0.5, maximum=1.0, step=0.05, value=0.7,
                        scale=1,
                    )

                export_author = gr.Radio(
                    label="Author filter",
                    choices=["all", "human only", "claude only"],
                    value="all",
                )
                export_btn = gr.Button("Export to CSV", variant="primary")
                export_result = gr.HTML("")

                def do_export(path, min_conf, author_filter):
                    author = None
                    if author_filter == "human only":
                        author = "human"
                    elif author_filter == "claude only":
                        author = "claude"

                    try:
                        result = db.export_approved(path, min_confidence=min_conf, author_filter=author)
                    except Exception as e:
                        return f'<p style="color:#f38ba8">✗ Export failed: {e}</p>'

                    if result["exported"] == 0:
                        return (
                            '<p style="color:#f9e2af">⚠ No approved decisions matched the filters. '
                            'Run more curation first.</p>'
                        )

                    pos_pct = result["pos"] / result["exported"] if result["exported"] else 0
                    next_step = (
                        f"python classifier/scripts/build_v12_dataset.py \\\n"
                        f"  --extra-sources {result['path']} \\\n"
                        f"  --output classifier/data/training/training_data_v13b.csv"
                    )

                    return f"""
<div style="background:#1e1e2e;border:1px solid #a6e3a1;border-radius:10px;padding:20px;">
  <p style="color:#a6e3a1;font-size:1.1rem;font-weight:700">✓ Export complete</p>
  <p style="color:#cdd6f4">
    <strong>{result['exported']:,}</strong> sentences exported
    &nbsp;·&nbsp;
    <span style="color:#a6e3a1">{result['pos']:,} positive</span>
    &nbsp;·&nbsp;
    <span style="color:#f38ba8">{result['neg']:,} negative</span>
    &nbsp;·&nbsp;
    <span style="color:#a6adc8">({pos_pct:.1%} pos)</span>
  </p>
  <p style="color:#6c7086;font-size:0.85rem">→ {result['path']}</p>
  <hr style="border-color:#313244;margin:12px 0">
  <p style="color:#a6adc8;font-size:0.85rem">Next step:</p>
  <pre style="background:#181825;padding:12px;border-radius:6px;
              color:#cba6f7;font-size:0.82rem;overflow-x:auto">{next_step}</pre>
</div>
"""

                export_btn.click(
                    fn=do_export,
                    inputs=[export_path, export_min_conf, export_author],
                    outputs=export_result,
                )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaP Curation Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    parser.add_argument("--source", default="", help="Default source filter")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link")
    args = parser.parse_args()

    print(f"\n  MetaP Curation Interface → http://localhost:{args.port}\n")
    app = build_app(default_source=args.source)
    theme = gr.themes.Default(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, theme=theme, css=CUSTOM_CSS)
