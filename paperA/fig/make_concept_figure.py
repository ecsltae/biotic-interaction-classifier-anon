#!/usr/bin/env python3
"""Figure 1 (fig:concept): one passage, several possible pairs, two questions, three inputs.

Top band: the passage of the benchmark item whose sentence contains "Lone Star tick" (clean index
351 of clean_benchmark(); block Test299; candidate (Anaplasma, transmits, Ehrlichia); gold
negative), read from the benchmark file and drawn verbatim. Scientific names are italic and the
vernacular "Lone Star tick" stays roman, as biologists write them; the tick's two names count as
one organism and are kept on one line; the interaction term is bold.

Middle band: the three pairs of organisms the passage names; the single answer the sentence
question gives all of them (one merged cell); the pair question's answer for each pair; and the
correct answer. Only the third pair is a benchmark item, and its "no" comes from the gold label;
the first two pairs are illustrative and have no model score.

Bottom band: the three models' inputs, as the paper defines them: the sentence-level model reads
the passage alone; the pair-conditioned model reads s1 [SEP] s2 as segment A and the passage as
segment B; the triple-conditioned model reads s1 [SEP] r [SEP] s2 as segment A.

No model score is plotted. The script prints the three-checkpoint ensembles' scores for the
benchmark item against their block-held-out thresholds (the caption says the sentence-level model
accepts it and the pair- and triple-conditioned models reject it), checks every value against the
blueprint, and writes the values to fig_concept.json next to the figure. Before trusting index
351 it recomputes each model's pooled F1 and acceptance count from the score vectors and the
benchmark labels and compares them with tables_biodiv.json, so a misaligned score file cannot
pass. Any mismatch exits with status 1 before anything is written.

Style: Okabe-Ito colours as in make_threshold_figure.py; serif text at 7 to 7.5 pt; 3.03 x 2.7 in,
the column width, so include it at the column width without rescaling; TrueType fonts
(pdf.fonttype 42), no Type 3; no creator, producer, author or title metadata.

Usage
  python3 paperA/fig/make_concept_figure.py
  python3 paperA/fig/make_concept_figure.py --labels current --results-dir <dir> --out-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3: the ACL checker flags Type 3
plt.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, clean_benchmark, label_version  # noqa: E402

RESULTS = REPO / "results/paperA_v2"
FIGDIR = Path(__file__).resolve().parent
STEM = "fig_concept"

# Okabe-Ito, as in make_threshold_figure.py, plus the shared neutrals of the paper's figures
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
MUTED, INK = "#55554f", "#1a1a19"
PANEL, PANEL_EDGE = "#f4f4f1", "#cfcfc9"

W, H = 3.03, 2.7          # inches; the axes spans the figure, so data units are inches
MARGIN = 0.03
PT_PASSAGE, PT_BODY, PT_HEAD = 7.5, 7.2, 7.0   # nothing smaller than 7 pt
PITCH_HEAD = 0.118        # line pitch of the multi-line column headers, inches

# The item, as the blueprint fixes it. Everything except the organism styling is checked against
# the benchmark row, not taken from here.
ITEM = {"needle": "Lone Star tick", "clean_index": 351, "label": 0, "source": "test299",
        "species1": "Anaplasma", "relation": "transmits", "species2": "Ehrlichia"}
TICK = "Amblyomma americanum"      # the third organism the passage names (Lone Star tick)
# Values the blueprint states for the item, at the precision it prints them.
EXPECTED = {"sentence": (0.424, 0.04, "accept"), "pair": (0.007, 0.15, "reject"),
            "triple": (0.006, 0.10, "reject")}
ARM_NAME = {"sentence": "sentence-level", "pair": "pair-conditioned", "triple": "triple-conditioned"}

# Serif faces in order of preference. The figure must embed TrueType: matplotlib writes the
# CFF-flavoured OpenType files of Nimbus Roman and TeX Gyre Termes (.otf) under a TrueType font
# type, which PDF readers flag as a mismatch, so a family is used only if its faces are .ttf.
# Liberation Serif is metric-compatible with Times New Roman.
SERIF_PREFERENCE = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes",
                    "DejaVu Serif"]
FACES = {"normal": ("normal", "normal"), "italic": ("italic", "normal"), "bold": ("normal", "bold")}


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


# Fonts and text measurement.
def _faces(family: str) -> dict | None:
    """The regular, italic and bold files of `family` if all three are distinct TrueType files."""
    paths = {}
    for name, (style, weight) in FACES.items():
        try:
            p = fm.findfont(fm.FontProperties(family=family, style=style, weight=weight),
                            fallback_to_default=False)
        except ValueError:
            return None
        paths[name] = p
    if len(set(paths.values())) < 3 or not all(p.lower().endswith(".ttf") for p in paths.values()):
        return None
    return paths


def _register_system_ttf(family: str) -> None:
    """Add the system's .ttf files of `family` to matplotlib's list (its cache may predate them)."""
    for path in fm.findSystemFonts(fontext="ttf"):
        if not path.lower().endswith(".ttf"):
            continue
        try:
            if FT2Font(path).family_name == family:
                fm.fontManager.addfont(path)
        except (OSError, RuntimeError):
            continue


def pick_serif() -> tuple[str, dict]:
    for family in SERIF_PREFERENCE:
        faces = _faces(family)
        if faces is None:
            _register_system_ttf(family)
            faces = _faces(family)
        if faces is not None:
            return family, faces
    fail("no TrueType serif family found")


class Fonts:
    def __init__(self, family: str):
        self.family = family

    def __call__(self, style: str = "normal", size: float = PT_BODY) -> fm.FontProperties:
        st, wt = FACES[style]
        return fm.FontProperties(family=self.family, style=st, weight=wt, size=size)


_TTP = TextToPath()


def width_in(text: str, fp: fm.FontProperties) -> float:
    """Advance width of `text` in inches, spaces included (unhinted outline metrics)."""
    a = _TTP.get_text_width_height_descent("|" + text + "|", fp, ismath=False)[0]
    b = _TTP.get_text_width_height_descent("||", fp, ismath=False)[0]
    return (a - b) / 72.0


def em(size: float) -> float:
    return size / 72.0


# Styled text: runs of (text, style) set on one baseline.
def style_runs(text: str, italic=(), bold=()) -> list:
    """Split `text` into (piece, style) runs: the `italic` terms italic, the `bold` terms bold."""
    marks = []
    for terms, style in ((italic, "italic"), (bold, "bold")):
        for term in terms:
            start = text.find(term)
            check(start >= 0, f"{term!r} does not occur in {text!r}")
            while start >= 0:
                marks.append((start, start + len(term), style))
                start = text.find(term, start + len(term))
    marks.sort()
    runs, pos = [], 0
    for a, b, style in marks:
        check(a >= pos, f"styled terms overlap in {text!r}")
        if a > pos:
            runs.append((text[pos:a], "normal"))
        runs.append((text[a:b], style))
        pos = b
    if pos < len(text):
        runs.append((text[pos:], "normal"))
    check("".join(p for p, _ in runs) == text, "styled runs do not reproduce the text")
    return runs


def words_of(runs: list) -> list:
    """Split runs at spaces into words, each a list of (piece, style)."""
    words, cur = [], []
    for piece, style in runs:
        for i, part in enumerate(piece.split(" ")):
            if i > 0 and cur:
                words.append(cur)
                cur = []
            if part:
                cur.append((part, style))
    if cur:
        words.append(cur)
    return words


def wrap(words: list, max_w: float, F: Fonts, size: float) -> list:
    """Greedy line breaking; returns lists of words."""
    space = width_in(" ", F("normal", size))
    lines, cur, cur_w = [], [], 0.0
    for w in words:
        ww = sum(width_in(p, F(s, size)) for p, s in w)
        if cur and cur_w + space + ww > max_w:
            lines.append(cur)
            cur, cur_w = [w], ww
        else:
            cur_w += (space if cur else 0.0) + ww
            cur.append(w)
    lines.append(cur)
    return lines


def line_runs(line: list) -> list:
    """Join a line's words with spaces and merge neighbouring pieces of the same style."""
    runs = []
    for i, w in enumerate(line):
        pieces = list(w)
        if i < len(line) - 1:
            p, s = pieces[-1]
            pieces[-1] = (p + " ", s)
        for p, s in pieces:
            if runs and runs[-1][1] == s:
                runs[-1] = (runs[-1][0] + p, s)
            else:
                runs.append((p, s))
    return runs


def runs_width(runs: list, F: Fonts, size: float) -> float:
    return sum(width_in(p, F(s, size)) for p, s in runs)


def draw_runs(ax, x: float, y: float, runs: list, F: Fonts, size: float,
              color: str = INK) -> float:
    """Draw runs left to right on the baseline y, starting at x; return the end x."""
    for p, s in runs:
        fp = F(s, size)
        ax.text(x, y, p, fontproperties=fp, color=color, ha="left", va="baseline")
        x += width_in(p, fp)
    return x


def box(ax, x: float, y: float, w: float, h: float, ec: str, fc: str = "white", lw: float = 0.9,
        r: float = 0.03) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=1))


# Data.
def load_item(labels: str, results: Path) -> dict:
    d = clean_benchmark(labels)
    v = label_version(labels)
    tables = json.loads((results / "tables_biodiv.json").read_text())
    check(tables.get("labels") == labels, f"tables_biodiv.json was built for labels "
                                          f"{tables.get('labels')!r}, not {labels!r}")
    check(tables.get("benchmark_sha256") == v["sha256"], "tables_biodiv.json benchmark hash differs")
    check(len(d) == tables["n"] and int(d.label.sum()) == tables["positives"],
          "benchmark size or positives differ from tables_biodiv.json")

    hits = d.index[d.sentence.str.contains(ITEM["needle"], regex=False)].tolist()
    check(hits == [ITEM["clean_index"]], f"rows containing {ITEM['needle']!r}: {hits}, expected "
                                         f"[{ITEM['clean_index']}]")
    i = ITEM["clean_index"]
    row = d.loc[i]
    for k in ("label", "source", "species1", "relation", "species2"):
        check(row[k] == ITEM[k], f"row {i} {k} = {row[k]!r}, expected {ITEM[k]!r}")
    y = d.label.to_numpy()
    blocks = d.source.to_numpy()

    arms = {}
    for arm in ("sentence", "pair", "triple"):
        S = np.load(results / f"S_biodiv_{arm}.npy")
        check(S.shape == (len(d),), f"S_biodiv_{arm}.npy has shape {S.shape}")
        thr = tables[arm]["thresholds"]
        # row alignment: the block-held-out decisions must reproduce the table's pooled numbers
        pred = np.array([int(s >= thr[b]) for s, b in zip(S, blocks)])
        f1 = f1_score(y, pred)
        check(abs(f1 - tables[arm]["F1"]) < 1e-9 and int(pred.sum()) == tables[arm]["accepts"],
              f"{arm}: F1 {f1:.4f} / {pred.sum()} accepts recomputed, table has "
              f"{tables[arm]['F1']:.4f} / {tables[arm]['accepts']}")
        score, t = float(S[i]), float(thr[row.source])
        decision = "accept" if score >= t else "reject"   # the rule of rebuild_paperA_numbers.py
        exp_score, exp_t, exp_dec = EXPECTED[arm]
        check(round(score, 3) == exp_score, f"{arm} score {score:.4f}, blueprint {exp_score}")
        check(round(t, 2) == exp_t, f"{arm} threshold {t:.4f}, blueprint {exp_t}")
        check(decision == exp_dec, f"{arm} {decision}s the item, blueprint says {exp_dec}")
        arms[arm] = {"score": score, "threshold": t, "decision": decision,
                     "f1_check": f1, "accepts_check": int(pred.sum())}
    return {"row": row, "arms": arms, "provenance": {"labels": labels, "sha256": v["sha256"],
                                                    "n": len(d), "positives": int(y.sum())}}


# Figure.
def abbreviate(binomial: str) -> str:
    genus, epithet = binomial.split(" ", 1)
    return f"{genus[0]}. {epithet}"


def draw(item: dict, F: Fonts):
    """Draw the figure; return it, the layout (inches) and the rows of the middle band."""
    row = item["row"]
    s1, rel, s2 = row.species1, row.relation, row.species2
    passage = row.sentence
    check(passage.count(TICK) == 1, f"{TICK!r} does not occur once in the passage")
    tick = abbreviate(TICK)
    gold = "yes" if int(row.label) == 1 else "no"

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cap = 0.662          # cap height of a Times-metric face, in em
    desc = 0.216         # descender depth, in em

    # Top band: the passage.
    top = H - MARGIN
    pad = 0.065
    y = top - 0.052 - cap * em(PT_HEAD)
    ax.text(MARGIN + pad, y, "Passage", fontproperties=F("normal", PT_HEAD), color=MUTED,
            ha="left", va="baseline")
    runs = style_runs(passage, italic=(TICK, s1, s2), bold=(rel,))
    lines = wrap(words_of(runs), W - 2 * MARGIN - 2 * pad, F, PT_PASSAGE)
    check(len(lines) == 2, f"the passage wraps to {len(lines)} lines, expected 2")
    check(" ".join("".join(p for p, _ in w) for ln in lines for w in ln) == passage,
          "the drawn passage differs from the benchmark sentence")
    check(any(any(p.startswith("Lone") for p, _ in w) for w in lines[0])
          and any(any(p.startswith("americanum") for p, _ in w) for w in lines[0]),
          "the tick's two names are split across lines")
    y -= 0.16
    for ln in lines:
        draw_runs(ax, MARGIN + pad, y, line_runs(ln), F, PT_PASSAGE)
        y -= 0.145
    y += 0.145
    bottom = y - desc * em(PT_PASSAGE) - 0.06
    box(ax, MARGIN, bottom, W - 2 * MARGIN, top - bottom, ec=PANEL_EDGE, fc=PANEL, lw=0.6, r=0.05)

    fh = F("normal", PT_HEAD)

    def header_row(y_top: float, cols: list) -> float:
        """Bottom-aligned muted headers; cols = (lines, x, ha). Returns the rule's height."""
        n = max(len(lines) for lines, _, _ in cols)
        base = y_top - cap * em(PT_HEAD) - (n - 1) * PITCH_HEAD
        for lines, cx, ha in cols:
            for j, t in enumerate(reversed(lines)):
                ax.text(cx, base + j * PITCH_HEAD, t, fontproperties=fh, color=MUTED, ha=ha,
                        va="baseline")
        rule = base - desc * em(PT_HEAD) - 0.05
        ax.plot([MARGIN, W - MARGIN], [rule, rule], color=PANEL_EDGE, lw=0.6,
                solid_capstyle="butt")
        return rule

    # Middle band: possible pairs and the answers to the two questions.
    pairs = [((tick, s1), "yes", "yes"), ((tick, s2), "yes", "yes"), ((s1, s2), gold, gold)]
    check([(pq, cor) for _, pq, cor in pairs] == [("yes", "yes"), ("yes", "yes"), ("no", "no")],
          "pair-question and correct answers differ from the blueprint's yes, yes, no")
    pair_runs = [[(a, "italic"), (", ", "normal"), (b, "italic")] for (a, b), _, _ in pairs]
    head = {
        "pairs": ["Possible pairs"],
        "sent": ["Sentence question:", "does the passage", "describe an interaction?"],
        "pair": ["Pair question:", "do these two", "interact?"],
        "gold": ["Gold"],
    }
    hw = {k: max(width_in(t, fh) for t in v) for k, v in head.items()}
    cell_w, cell_h, gap = 0.42, 0.18, 0.12
    x0 = MARGIN + 0.005
    c_gold = W - MARGIN - hw["gold"] / 2 - 0.01
    c_pair = c_gold - hw["gold"] / 2 - gap - hw["pair"] / 2
    c_sent = c_pair - hw["pair"] / 2 - gap - hw["sent"] / 2
    pairs_end = x0 + max(runs_width(r, F, PT_BODY) for r in pair_runs)
    check(c_sent - cell_w / 2 - pairs_end > 0.08, "the pair labels run into the sentence column")

    rule = header_row(bottom - 0.115, [(head["pairs"], x0, "left"), (head["sent"], c_sent, "center"),
                                      (head["pair"], c_pair, "center"),
                                      (head["gold"], c_gold, "center")])
    pitch_r = 0.228
    rows_y = [rule - 0.16 - k * pitch_r for k in range(3)]      # row centres
    shift = 0.3 * em(PT_BODY)                                    # centre line to baseline
    for k, ((_, pq, cor), runs_k) in enumerate(zip(pairs, pair_runs)):
        yb = rows_y[k] - shift
        draw_runs(ax, x0, yb, runs_k, F, PT_BODY)
        box(ax, c_pair - cell_w / 2, rows_y[k] - cell_h / 2, cell_w, cell_h, ec=BLUE)
        ax.text(c_pair, yb, pq, fontproperties=F("normal", PT_BODY), color=INK, ha="center",
                va="baseline", zorder=2)
        ax.text(c_gold, yb, cor, fontproperties=F("normal", PT_BODY), color=INK, ha="center",
                va="baseline")
    # the sentence question: one cell, one answer, whichever pair the candidate names
    top_c, bot_c = rows_y[0] + cell_h / 2, rows_y[-1] - cell_h / 2
    box(ax, c_sent - cell_w / 2, bot_c, cell_w, top_c - bot_c, ec=VERM)
    ax.text(c_sent, rows_y[1] - shift, "yes", fontproperties=F("normal", PT_BODY), color=INK,
            ha="center", va="baseline", zorder=2)
    # ... which is wrong for the third pair
    x_cross = c_sent + cell_w / 2 + 0.085
    check(x_cross + 0.05 < c_pair - cell_w / 2, "the cross runs into the pair column")
    ax.plot([x_cross], [rows_y[-1]], ls="none", marker="x", ms=5.5, mew=1.3, color=VERM,
            zorder=3)

    # Bottom band: the three models' inputs (segment A, then the passage).
    fi = F("normal", PT_HEAD)
    label_w = max(width_in(ARM_NAME[a], fi) for a in ARM_NAME)
    x_in = x0 + label_w + 0.045
    bpad, bgap, bh = 0.03, 0.03, 0.16
    sep = [("[SEP]", "normal")]
    sp = [(" ", "normal")]
    queries = {"sentence": None,
               "pair": [(s1, "italic")] + sp + sep + sp + [(s2, "italic")],
               "triple": [(s1, "italic")] + sp + sep + sp + [(rel, "bold")] + sp + sep + sp
                         + [(s2, "italic")]}
    edge = {"pair": BLUE, "triple": GREEN}
    rule_in = header_row(bot_c - 0.175, [(["Model"], x0, "left"), (["Input"], x_in, "left")])
    in_pitch = 0.205
    rows_in = [rule_in - 0.135 - k * in_pitch for k in range(3)]
    shift_in = 0.3 * em(PT_HEAD)
    right_end = 0.0
    for k, arm in enumerate(("sentence", "pair", "triple")):
        yc = rows_in[k]
        yb = yc - shift_in
        ax.text(x0, yb, ARM_NAME[arm], fontproperties=fi, color=INK, ha="left", va="baseline")
        x = x_in
        q = queries[arm]
        if q is not None:
            qw = runs_width(q, F, PT_HEAD) + 2 * bpad
            box(ax, x, yc - bh / 2, qw, bh, ec=edge[arm], r=0.025)
            # [SEP] is a token, not a word: set it in the muted ink
            xx = x + bpad
            for p, s in q:
                fp = F(s, PT_HEAD)
                ax.text(xx, yb, p, fontproperties=fp, color=MUTED if p == "[SEP]" else INK,
                        ha="left", va="baseline", zorder=2)
                xx += width_in(p, fp)
            x += qw + bgap
        pw = width_in("passage", fi) + 2 * bpad
        box(ax, x, yc - bh / 2, pw, bh, ec=PANEL_EDGE, fc=PANEL, lw=0.6, r=0.025)
        ax.text(x + pw / 2, yb, "passage", fontproperties=fi, color=MUTED, ha="center",
                va="baseline", zorder=2)
        right_end = max(right_end, x + pw)
    check(right_end <= W - MARGIN + 1e-9, f"the input strip ends at {right_end:.3f} in, past the "
                                          f"right margin")
    low = rows_in[-1] - bh / 2
    check(MARGIN - 0.012 <= low <= MARGIN + 0.05,
          f"the figure ends {low:.3f} in above the bottom edge; respace the bands")
    layout = {"passage_panel": [round(bottom, 3), round(top, 3)], "rule": round(rule, 3),
              "input_rule": round(rule_in, 3), "bottom_margin": round(low, 3),
              "rows": [round(v, 3) for v in rows_y], "inputs": [round(v, 3) for v in rows_in],
              "input_strip_right": round(right_end, 3),
              "columns": {"sentence": round(c_sent, 3), "pair": round(c_pair, 3),
                          "correct": round(c_gold, 3)}}
    return fig, layout, pairs


def check_text_layout(fig) -> int:
    """Every text inside the canvas, and no two texts overlapping (beyond italic overhang)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in texts]
    for t, b in zip(texts, boxes):
        check(b.x0 >= canvas.x0 - 0.5 and b.x1 <= canvas.x1 + 0.5 and b.y0 >= canvas.y0 - 0.5
              and b.y1 <= canvas.y1 + 0.5, f"text {t.get_text()!r} is cut off at the canvas edge")
    tol = 0.02 * fig.dpi        # 0.02 in: neighbouring runs of one line may touch
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            check(not (ox > tol and oy > tol),
                  f"texts {texts[i].get_text()!r} and {texts[j].get_text()!r} overlap")
    return len(texts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current")
    ap.add_argument("--results-dir", default=str(RESULTS),
                    help="where S_biodiv_*.npy and tables_biodiv.json are")
    ap.add_argument("--out-dir", default=str(FIGDIR), help="where the figure and its JSON go")
    a = ap.parse_args()
    results, out_dir = Path(a.results_dir), Path(a.out_dir)

    family, faces = pick_serif()
    F = Fonts(family)
    item = load_item(a.labels, results)
    row, arms = item["row"], item["arms"]
    i = ITEM["clean_index"]
    print(f"benchmark item: clean index {i}, block {row.source}, ({row.species1}, {row.relation}, "
          f"{row.species2}), gold {int(row.label)}")
    for arm, v in arms.items():
        op = ">=" if v["decision"] == "accept" else "<"
        print(f"  {ARM_NAME[arm]:<19} S_biodiv_{arm}.npy[{i}] = {v['score']:.3f} {op} "
              f"{arm}.thresholds.{row.source} = {v['threshold']:.2f}: {v['decision']}"
              f"   (row check: F1 {v['f1_check']:.3f}, {v['accepts_check']} accepts)")

    fig, layout, pairs = draw(item, F)
    n_texts = check_text_layout(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": None, "Producer": None, "CreationDate": None, "Author": None,
            "Title": None}
    fig.savefig(out_dir / f"{STEM}.pdf", metadata=meta)
    fig.savefig(out_dir / f"{STEM}.png", dpi=300,
                metadata={"Software": None, "Creator": None, "Author": None, "Title": None})
    plt.close(fig)

    record = {
        "figure": f"{STEM}.pdf",
        "latex_label": "fig:concept",
        "generator": Path(__file__).name,
        "benchmark": item["provenance"],
        "item": {"clean_index": i, "source": row.source, "sentence": row.sentence,
                 "species1": row.species1, "relation": row.relation, "species2": row.species2,
                 "label": int(row.label)},
        # keys as the blueprint names them; raw values from the result files
        "values": {**{f"S_biodiv_{arm}.npy[{i}]": v["score"] for arm, v in arms.items()},
                   **{f"{arm}.thresholds.{row.source}": v["threshold"] for arm, v in arms.items()}},
        "decisions": {arm: {"model": ARM_NAME[arm], "score": round(v["score"], 3),
                            "threshold": round(v["threshold"], 2),
                            "rule": "accept if score >= threshold (block-held-out threshold of the "
                                    "item's block; three-checkpoint ensemble)",
                            "decision": v["decision"]} for arm, v in arms.items()},
        "possible_pairs": [
            {"row": k + 1, "pair": f"{a}, {b}", "benchmark_item": k == 2,
             "sentence_question": "yes", "pair_question": pq, "correct": cor,
             "model_scores": ({arm: round(v["score"], 3) for arm, v in arms.items()}
                              if k == 2 else None)}
            for k, ((a, b), pq, cor) in enumerate(pairs)],
        "row_alignment_check": {arm: {"F1": round(v["f1_check"], 4), "accepts": v["accepts_check"]}
                                for arm, v in arms.items()},
        "font": {"family": family, "files": {k: Path(p).name for k, p in faces.items()}},
        "size_in": [W, H],
        "layout_in": layout,
    }
    (out_dir / f"{STEM}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"font: {family} ({', '.join(Path(p).name for p in faces.values())}); "
          f"{n_texts} text objects inside the canvas, none overlapping")
    print(f"wrote {out_dir / STEM}.{{pdf,png,json}}")


if __name__ == "__main__":
    main()
