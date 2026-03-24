#!/usr/bin/env python3
"""
根据 SQLite chunks 表补全黄金测试集中的 expected_content_checks。

- 校验 golden_chunk_ids 在对应 collection 中存在
- min_citations: 固定为 1（检索命中时至少一条引用）
- keywords_in_markdown: 从 chunk 文本前 N 字截取稳定子串（与 build_mcp_content 的 max_chars_per_chunk 对齐）
- expect_images: metadata 中 image_refs 非空时为 true

用法:
  python scripts/enrich_golden_content_checks.py
  python scripts/enrich_golden_content_checks.py --golden-set path.json --sqlite path.sqlite
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 须与 response_builder.build_mcp_content 默认一致
DEFAULT_MAX_CHUNK_CHARS = 1000


def _parse_metadata(mj: Optional[str]) -> Dict[str, Any]:
    if not mj:
        return {}
    try:
        return json.loads(mj)
    except json.JSONDecodeError:
        return {}


def _pick_keywords(text: str, max_chars: int) -> List[str]:
    if not text or not str(text).strip():
        return []
    t = str(text).strip()[:max_chars]
    kws: List[str] = []
    for line in t.splitlines():
        s = line.strip()
        if len(s) < 8:
            continue
        if s.startswith("#") and len(s) < 20:
            continue
        frag = s[: min(36, len(s))]
        if len(frag) >= 8:
            kws.append(frag)
            break
    if not kws:
        collapsed = re.sub(r"\s+", " ", t).strip()
        if len(collapsed) >= 10:
            kws.append(collapsed[: min(32, len(collapsed))])
    if len(t) > 200:
        mid = min(len(t) // 2, len(t) - 30)
        mid = max(30, mid)
        window = t[mid : mid + 40]
        line = window.split("\n")[0].strip()
        if len(line) >= 8:
            frag2 = line[: min(28, len(line))]
            if frag2 not in kws[0] and kws[0][:12] not in frag2:
                kws.append(frag2)
    return kws[:2]


def _fetch_chunk(
    conn: sqlite3.Connection, chunk_id: str, collection: str
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT text, metadata_json FROM chunks WHERE id = ? AND collection_name = ?",
        (chunk_id, collection),
    ).fetchone()
    if not row:
        return None
    return {"text": row[0], "metadata_json": row[1]}


def enrich(
    cases: List[Dict[str, Any]],
    sqlite_path: Path,
    max_chunk_chars: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]]]:
    conn = sqlite3.connect(sqlite_path)
    missing: List[Tuple[str, str, str]] = []

    for item in cases:
        coll = item.get("collection") or "cslearning"
        gids = item.get("golden_chunk_ids") or []
        primary = gids[0] if gids else None
        checks: Dict[str, Any] = {"min_citations": 1}

        if not primary:
            item["expected_content_checks"] = checks
            continue

        row = _fetch_chunk(conn, primary, coll)
        if not row:
            missing.append((item.get("query", "")[:40], primary, coll))
            item["expected_content_checks"] = checks
            continue

        text = row["text"] or ""
        meta = _parse_metadata(row["metadata_json"])
        refs = meta.get("image_refs") or []
        if isinstance(refs, list) and len(refs) > 0:
            checks["expect_images"] = True

        kws = _pick_keywords(text, max_chunk_chars)
        if kws:
            checks["keywords_in_markdown"] = kws

        item["expected_content_checks"] = checks

    conn.close()
    return cases, missing


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="补全黄金集 expected_content_checks")
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=root / "tests/fixtures/golden_test_set.json",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=root / "data/db/rag.sqlite",
        help="chunks 表所在 SQLite（默认 data/db/rag.sqlite）",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help="与 build_mcp_content max_chars_per_chunk 一致，用于截取关键词",
    )
    args = parser.parse_args()

    if not args.golden_set.is_file():
        print(f"找不到黄金集: {args.golden_set}", file=sys.stderr)
        return 1
    if not args.sqlite.is_file():
        print(f"找不到数据库: {args.sqlite}", file=sys.stderr)
        return 1

    data = json.loads(args.golden_set.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("黄金集须为 JSON 数组", file=sys.stderr)
        return 1

    enriched, missing = enrich(data, args.sqlite, args.max_chunk_chars)
    args.golden_set.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"已写入 {args.golden_set}，共 {len(enriched)} 条")
    if missing:
        print(f"警告: {len(missing)} 条在库中未找到 chunk（仅保留 min_citations）:")
        for q, cid, coll in missing[:20]:
            print(f"  [{coll}] {cid} query={q!r}...")
        if len(missing) > 20:
            print(f"  ... 另有 {len(missing) - 20} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
