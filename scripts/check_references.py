#!/usr/bin/env python3
"""Look every cited reference up on Semantic Scholar: does a paper with this title exist, with this
first author and year? ARR desk-rejects papers with inaccurate or invented references.

Reports, per entry: OK (title, first author and year agree), CHECK (found, but a field differs),
or NOT FOUND (needs a manual look; software, datasets and reports are often absent).

Usage
  python3 scripts/check_references.py paperA/references.bib paperA/paperA.aux
"""
from __future__ import annotations

import difflib
import re
import sys
import time
import unicodedata

import requests


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def entries(bib: str):
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", bib, re.S):
        typ, key, body = m.groups()
        f = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*[{\"]((?:[^{}]|\{[^{}]*\})*)[}\"]", body)}
        yield typ.lower(), key, f


def first_surname(authors: str) -> str:
    a = authors.split(" and ")[0].strip()
    if not a:
        return ""
    a = a.split(",")[0] if "," in a else a.split()[-1]
    return norm(a)


def main() -> None:
    bib = open(sys.argv[1], encoding="utf-8").read()
    cited = set()
    if len(sys.argv) > 2:
        cited = set(k for c in re.findall(r"\\citation\{([^}]*)\}", open(sys.argv[2]).read()) for k in c.split(","))
    rows = []
    for typ, key, f in entries(bib):
        if cited and key not in cited:
            continue
        title = f.get("title", ""); year = f.get("year", ""); first = first_surname(f.get("author", ""))
        q = re.sub(r"[{}]", "", title)
        hit = None
        for attempt in range(4):
            r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/match",
                             params={"query": q, "fields": "title,year,authors,venue"}, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            if r.ok and r.json().get("data"):
                hit = r.json()["data"][0]
            break
        time.sleep(1.2)
        if not hit:
            rows.append((key, "NOT FOUND", title[:70], "")); continue
        tsim = difflib.SequenceMatcher(None, norm(q), norm(hit["title"])).ratio()
        a1 = norm(hit["authors"][0]["name"].split()[-1]) if hit.get("authors") else ""
        issues = []
        if tsim < 0.9: issues.append(f"title~{tsim:.2f}: {hit['title'][:60]}")
        if first and a1 and first not in a1 and a1 not in first and "others" not in first: issues.append(f"first author {a1} vs {first}")
        if year and hit.get("year") and abs(int(re.sub(r'\D', '', year) or 0) - int(hit["year"])) > 1: issues.append(f"year {hit['year']} vs {year}")
        rows.append((key, "OK" if not issues else "CHECK", title[:70], "; ".join(issues)))
    for k, st, t, why in rows:
        print(f"{st:9} {k:40} {t}" + (f"\n          -> {why}" if why else ""))
    print(f"\n{sum(r[1] == 'OK' for r in rows)} OK, {sum(r[1] == 'CHECK' for r in rows)} CHECK, "
          f"{sum(r[1] == 'NOT FOUND' for r in rows)} NOT FOUND, of {len(rows)} cited entries")


if __name__ == "__main__":
    main()
