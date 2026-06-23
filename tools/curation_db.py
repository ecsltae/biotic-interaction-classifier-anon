"""
Curation Database — SQLite-backed queue for LLM + human validation of training sentences.

Provides a persistent store for the curation workflow:
  import_csv() → get_pending() → submit_decision() → export_approved()

DB path: classifier/data/training/curation.db  (git-ignored)
"""
from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "training" / "curation.db"

# Training CSV columns expected by build_v*_dataset.py
EXPORT_COLUMNS = ["text", "label", "interaction_type", "source_species", "target_species", "source"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS curation_queue (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    text             TEXT    NOT NULL,
    source           TEXT,
    orig_label       INTEGER DEFAULT -1,
    label            INTEGER,           -- NULL = pending decision
    confidence       REAL,
    reasoning        TEXT,
    author           TEXT,              -- 'claude' | 'human' | 'heuristic'
    status           TEXT    NOT NULL DEFAULT 'pending',
    interaction_type TEXT,
    source_species   TEXT,
    target_species   TEXT,
    heuristic_score  REAL,              -- pre-computed lexicon score for sorting
    source_file      TEXT,              -- original eval file name (e.g. eval_100.tsv)
    created_at       TEXT    NOT NULL,
    updated_at       TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_text ON curation_queue(text);
CREATE INDEX IF NOT EXISTS idx_source  ON curation_queue(source);
CREATE INDEX IF NOT EXISTS idx_status  ON curation_queue(status);
CREATE INDEX IF NOT EXISTS idx_author  ON curation_queue(author);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _conn(db_path: Path = DB_PATH):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create schema if not present."""
    with _conn(db_path) as con:
        con.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_csv(
    path: str | Path,
    source_tag: str,
    *,
    text_col: str = "text",
    label_col: str = "label",
    type_col: str = "interaction_type",
    sp1_col: str = "source_species",
    sp2_col: str = "target_species",
    source_file_col: str = "source_file",
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    """Load a CSV into the curation queue, skipping duplicate texts.

    Returns counts: imported, skipped_duplicates, pending_total.
    """
    init_db(db_path)
    path = Path(path)
    now = _now()
    imported = 0
    skipped = 0

    # Pre-compute heuristic score for smart ordering
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.data.interaction_lexicon import score_sentence
        _score = lambda t: score_sentence(t.lower())[1]  # noqa: E731
    except Exception:
        _score = lambda t: 0.0  # noqa: E731

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with _conn(db_path) as con:
        for row in rows:
            text = row.get(text_col, "").strip()
            if not text:
                continue
            orig_label = int(row.get(label_col, -1)) if row.get(label_col, "") != "" else -1
            try:
                con.execute(
                    """INSERT INTO curation_queue
                       (text, source, orig_label, interaction_type, source_species,
                        target_species, heuristic_score, source_file, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        text,
                        source_tag,
                        orig_label,
                        row.get(type_col, ""),
                        row.get(sp1_col, ""),
                        row.get(sp2_col, ""),
                        _score(text),
                        row.get(source_file_col, ""),
                        now,
                    ),
                )
                imported += 1
            except sqlite3.IntegrityError:
                skipped += 1

        total = con.execute("SELECT COUNT(*) FROM curation_queue WHERE status='pending'").fetchone()[0]

    return {"imported": imported, "skipped_duplicates": skipped, "pending_total": total}


# ---------------------------------------------------------------------------
# Queue retrieval
# ---------------------------------------------------------------------------

def get_pending(
    source: str | None = None,
    n: int = 20,
    *,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Return next N pending items, sorted by uncertainty first.

    Uncertainty = cases where heuristic_score disagrees with orig_label:
      - orig_label=1 but heuristic_score < 0.3  (possibly mislabeled positive)
      - orig_label=0 but heuristic_score > 0.6  (possibly mislabeled negative)
    These are most valuable to validate manually / with LLM.
    """
    init_db(db_path)
    where = "status = 'pending'"
    params: list[Any] = []
    if source:
        where += " AND source = ?"
        params.append(source)

    # Uncertainty score: abs(orig_label - heuristic_score) when orig_label != -1
    query = f"""
        SELECT id, text, source, orig_label, interaction_type,
               source_species, target_species, heuristic_score, source_file,
               ABS(COALESCE(orig_label, 0.5) - COALESCE(heuristic_score, 0.5)) AS uncertainty
        FROM curation_queue
        WHERE {where}
        ORDER BY uncertainty DESC, heuristic_score DESC
        LIMIT ?
    """
    params.append(n)

    with _conn(db_path) as con:
        rows = con.execute(query, params).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Decision submission
# ---------------------------------------------------------------------------

def submit_decision(
    item_id: int,
    label: int,
    confidence: float,
    reasoning: str,
    author: str,
    *,
    db_path: Path = DB_PATH,
) -> dict[str, Any]:
    """Save a curation decision for a queue item.

    Status mapping:
      confidence >= 0.7 → 'approved'
      0.4 <= confidence < 0.7 → 'uncertain'
      label == -1 (skip) → 'skip'
    """
    if label == -1:
        status = "skip"
    elif confidence >= 0.7:
        status = "approved"
    else:
        status = "uncertain"

    with _conn(db_path) as con:
        con.execute(
            """UPDATE curation_queue
               SET label=?, confidence=?, reasoning=?, author=?, status=?, updated_at=?
               WHERE id=?""",
            (label, confidence, reasoning, author, status, _now(), item_id),
        )
        pending = con.execute(
            "SELECT COUNT(*) FROM curation_queue WHERE status='pending'"
        ).fetchone()[0]

    return {"saved": True, "status": status, "pending_remaining": pending}


# ---------------------------------------------------------------------------
# Stats & review
# ---------------------------------------------------------------------------

def get_stats(db_path: Path = DB_PATH) -> dict[str, Any]:
    """Per-source breakdown + overall agreement rate."""
    init_db(db_path)
    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT source,
                      SUM(status='pending')   AS pending,
                      SUM(status='approved')  AS approved,
                      SUM(status='uncertain') AS uncertain,
                      SUM(status='skip')      AS skip,
                      COUNT(*)                AS total,
                      ROUND(AVG(CASE WHEN label IS NOT NULL THEN label END), 3) AS pos_rate
               FROM curation_queue
               GROUP BY source
               ORDER BY source"""
        ).fetchall()

        # Claude vs human agreement (rows annotated by both)
        agreement_rows = con.execute(
            """SELECT a.id,
                      a.label AS claude_label,
                      b.label AS human_label
               FROM curation_queue a
               JOIN curation_queue b ON a.id = b.id
               WHERE a.author = 'claude' AND b.author = 'human'"""
        ).fetchall()

    by_source = {}
    for r in rows:
        by_source[r["source"] or "unknown"] = {
            "pending":   r["pending"],
            "approved":  r["approved"],
            "uncertain": r["uncertain"],
            "skip":      r["skip"],
            "total":     r["total"],
            "pos_rate":  r["pos_rate"],
        }

    agreement = None
    if agreement_rows:
        matches = sum(1 for r in agreement_rows if r["claude_label"] == r["human_label"])
        agreement = round(matches / len(agreement_rows), 3)

    return {"by_source": by_source, "claude_human_agreement": agreement}


def list_decisions(
    status: str | None = None,
    author: str | None = None,
    source: str | None = None,
    uncertain_only: bool = False,
    *,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve decisions with optional filters."""
    init_db(db_path)
    conditions = []
    params: list[Any] = []

    if uncertain_only:
        conditions.append("status = 'uncertain'")
    elif status:
        conditions.append("status = ?")
        params.append(status)

    if author:
        conditions.append("author = ?")
        params.append(author)
    if source:
        conditions.append("source = ?")
        params.append(source)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"""
        SELECT id, text, source, orig_label, label, confidence,
               reasoning, author, status, interaction_type,
               source_species, target_species, source_file, updated_at
        FROM curation_queue {where}
        ORDER BY confidence ASC, updated_at DESC
    """

    with _conn(db_path) as con:
        rows = con.execute(query, params).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_approved(
    output_path: str | Path,
    min_confidence: float = 0.7,
    author_filter: str | None = None,
    *,
    db_path: Path = DB_PATH,
) -> dict[str, Any]:
    """Write approved decisions to a training-ready CSV.

    The CSV uses the same columns as training_data_v*.csv so it can be passed
    directly as --extra-sources to build_v*_dataset.py.
    """
    conditions = ["status = 'approved'", "label IS NOT NULL", "confidence >= ?"]
    params: list[Any] = [min_confidence]
    if author_filter:
        conditions.append("author = ?")
        params.append(author_filter)

    query = f"""
        SELECT text, label, interaction_type, source_species, target_species, source
        FROM curation_queue
        WHERE {" AND ".join(conditions)}
        ORDER BY confidence DESC
    """

    with _conn(db_path) as con:
        rows = con.execute(query, params).fetchall()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = neg = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
            if r["label"] == 1:
                pos += 1
            else:
                neg += 1

    return {
        "exported": pos + neg,
        "pos": pos,
        "neg": neg,
        "path": str(output_path),
        "min_confidence": min_confidence,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
