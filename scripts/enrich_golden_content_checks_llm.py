#!/usr/bin/env python3
"""
读取现有黄金测试集，按每条用例从 rag.sqlite 拉取 golden_chunk 正文与 metadata，
调用配置中的**大模型**判断并生成 expected_content_checks（非规则截断脚本）。

产出字段与 L2 评估器一致：
- min_citations: 一般为 1
- expect_images: 仅当 chunk metadata 含 image_refs 时应为 true
- keywords_in_markdown: 1～2 条，须为「与 MCP 展示一致」的 chunk 正文子串（前 1000 字内原文）

依赖：config/settings.yaml（及可选 settings.local.yaml）中 text LLM 可用。

用法（仓库根目录）:
  python scripts/enrich_golden_content_checks_llm.py
  python scripts/enrich_golden_content_checks_llm.py --golden-set tests/fixtures/golden_test_set.json
  python scripts/enrich_golden_content_checks_llm.py --dry-run --limit 3   # 不写文件、不改用例
  python scripts/enrich_golden_content_checks_llm.py --sleep 0.5
  python scripts/enrich_golden_content_checks_llm.py --limit 5 --out /tmp/partial.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 与 build_mcp_content(..., max_chars_per_chunk=1000) 对齐
MAX_CHARS_IN_MCP = 1000

_SYSTEM = """你是 RAG 评测数据标注员。你会收到：
- 检索问句 query
- 黄金 chunk_id
- 该 chunk 在 MCP 正文中会出现的前 N 个字符（N≤1000），字符串名为展示正文，必须逐字一致使用
- has_image_refs：该 chunk 元数据是否声明了图片引用

请只输出一个 JSON 对象（不要 markdown 代码块、不要其它文字），字段：
{
  "min_citations": 整数，检索命中并返回至少一条结果时通常为 1,
  "expect_images": 布尔，仅当 has_image_refs 为 true 时应为 true，否则 false,
  "keywords_in_markdown": 字符串数组，长度 1 或 2。每个字符串必须是「展示正文」中的连续子串，原样复制，长度建议 12～48 字符，
  用于 L2 检测：当该 chunk 被拼进 Markdown 时这些短语应出现。不要跨行拼接不存在的空格，不要改写标点。
  若展示正文过短可只给 1 条。不要照抄标题行若以 # 开头且信息量低，优先选实质陈述句中的片段。
}
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _parse_llm_checks(raw: str) -> Dict[str, Any]:
    s = _strip_json_fence(raw)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("顶层不是 JSON 对象")
    return obj


def _fetch_chunk(
    conn: sqlite3.Connection, chunk_id: str, collection: str
) -> Optional[Tuple[str, Dict[str, Any]]]:
    row = conn.execute(
        "SELECT text, metadata_json FROM chunks WHERE id = ? AND collection_name = ?",
        (chunk_id, collection),
    ).fetchone()
    if not row:
        return None
    text = row[0] or ""
    meta: Dict[str, Any] = {}
    if row[1]:
        try:
            meta = json.loads(row[1])
        except json.JSONDecodeError:
            meta = {}
    return text, meta


def _validate_and_trim_checks(
    checks: Dict[str, Any],
    display_text: str,
    has_image_refs: bool,
) -> Dict[str, Any]:
    """丢弃不在展示正文内的关键词；校正 expect_images。"""
    out: Dict[str, Any] = {"min_citations": 1}
    mc = checks.get("min_citations")
    if isinstance(mc, int) and mc >= 1:
        out["min_citations"] = mc

    exp = checks.get("expect_images")
    if has_image_refs:
        out["expect_images"] = bool(exp) if isinstance(exp, bool) else True
    else:
        out["expect_images"] = False
        if exp is True:
            logger.warning("模型要求 expect_images=true 但库中无 image_refs，已改为 false")

    kws = checks.get("keywords_in_markdown")
    good: List[str] = []
    if isinstance(kws, list):
        for k in kws:
            if isinstance(k, str) and len(k) >= 8 and k in display_text:
                good.append(k)
            elif isinstance(k, str):
                logger.debug("丢弃非子串关键词: %r", k[:60])
    if good:
        out["keywords_in_markdown"] = good[:2]
    return out


def _fallback_checks(display_text: str, has_image_refs: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"min_citations": 1}
    if has_image_refs:
        out["expect_images"] = True
    collapsed = " ".join(display_text.split())
    if len(collapsed) >= 12:
        frag = collapsed[: min(36, len(collapsed))]
        out["keywords_in_markdown"] = [frag]
    return out


def main() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(
        description="用大模型为黄金集补全 expected_content_checks",
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=root / "tests" / "fixtures" / "golden_test_set.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径，默认与 --golden-set 相同",
    )
    parser.add_argument("--db", type=Path, default=root / "data" / "db" / "rag.sqlite")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用 LLM，仅用数据库与简单 fallback 生成 checks",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0 表示全部")
    parser.add_argument("--sleep", type=float, default=0.0, help="每次 API 调用后休眠秒数")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单条 LLM 失败时用 fallback checks 并继续",
    )
    args = parser.parse_args()

    out_path = args.out or args.golden_set
    if args.dry_run:
        out_path = None  # 不写文件
    elif args.limit > 0 and args.out is None:
        out_path = args.golden_set.with_suffix(".partial.json")
        logger.warning(
            "已指定 --limit 且未指定 --out：写入旁路文件 %s（避免误覆盖完整黄金集）",
            out_path,
        )

    with open(args.golden_set, encoding="utf-8") as f:
        cases: List[Dict[str, Any]] = json.load(f)

    conn = sqlite3.connect(str(args.db))
    llm = None
    if not args.dry_run:
        from src.core.settings import load_settings
        from src.libs.llm.llm_factory import LLMFactory

        settings = load_settings(args.config)
        llm = LLMFactory.create(settings)
        logger.info("LLM: %s %s", llm.get_provider(), llm.get_model_name())

    n = len(cases) if args.limit <= 0 else min(args.limit, len(cases))

    for i in range(n):
        item = cases[i]
        gids = item.get("golden_chunk_ids") or []
        cid = gids[0] if gids else None
        coll = item.get("collection") or "cslearning"
        if not cid:
            logger.warning("[%d] 无 golden_chunk_ids，跳过", i)
            continue

        got = _fetch_chunk(conn, cid, coll)
        if not got:
            logger.error("[%d] 库中无 chunk %s (%s)", i, cid, coll)
            item["expected_content_checks"] = {"min_citations": 1}
            continue

        text, meta = got
        refs = meta.get("image_refs") or []
        has_img = isinstance(refs, list) and len(refs) > 0
        display = text.strip()[:MAX_CHARS_IN_MCP]

        if args.dry_run or llm is None:
            if not args.dry_run:
                item["expected_content_checks"] = _fallback_checks(display, has_img)
                logger.info("[%d] fallback: %s", i, cid)
            else:
                logger.info("[%d] dry-run 跳过写入: %s", i, cid)
            continue

        user_msg = (
            f"query: {item.get('query', '')}\n"
            f"chunk_id: {cid}\n"
            f"has_image_refs: {str(has_img).lower()}\n\n"
            "展示正文（请仅从下列字符中复制子串作为 keywords，不要增删改）：\n"
            f"{display}\n"
        )
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ]
            )
            parsed = _parse_llm_checks(raw)
            checks = _validate_and_trim_checks(parsed, display, has_img)
            if "keywords_in_markdown" not in checks:
                checks = _fallback_checks(display, has_img)
                logger.warning("[%d] LLM 未产出有效关键词，已 fallback: %s", i, cid)
            item["expected_content_checks"] = checks
            logger.info("[%d] OK %s keywords=%s", i, cid, checks.get("keywords_in_markdown"))
        except Exception as e:
            logger.exception("[%d] LLM 失败 %s: %s", i, cid, e)
            if args.continue_on_error:
                item["expected_content_checks"] = _fallback_checks(display, has_img)
            else:
                conn.close()
                raise SystemExit(1) from e

        if args.sleep > 0:
            time.sleep(args.sleep)

    conn.close()

    if out_path is None:
        print(f"Dry-run 完成：已处理 {n} 条（未写文件）")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}（共 {len(cases)} 条，本次更新前 {n} 条的 checks）")


if __name__ == "__main__":
    main()
