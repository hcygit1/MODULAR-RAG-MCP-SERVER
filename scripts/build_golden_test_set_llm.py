#!/usr/bin/env python3
"""
从 rag.sqlite 抽样 report / cslearning 的 chunk，用配置中的大模型阅读每段正文后生成检索问句，
写入黄金测试集 JSON（不使用规则模板拼接）。

依赖：config/settings.yaml（或 --config）中已配置可用的 text LLM（如 qwen / dashscope API key 等）。

用法（仓库根目录）:
  python scripts/build_golden_test_set_llm.py
  python scripts/build_golden_test_set_llm.py --out tests/fixtures/golden_test_set.json
  python scripts/build_golden_test_set_llm.py --dry-run   # 不调用 API，占位问句

补全 L2 的 expected_content_checks（由大模型生成，非规则截断）:
  python scripts/enrich_golden_content_checks_llm.py

仅重写已有黄金集中的 query（保留 golden_chunk_ids / collection / description / expected_content_checks）:
  python scripts/build_golden_test_set_llm.py --regenerate-queries-only
  python scripts/build_golden_test_set_llm.py --regenerate-queries-only --golden-set tests/fixtures/golden_test_set.json --out tests/fixtures/golden_test_set.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_chunk_pool_module():
    """加载 build_golden_test_set.py 以复用 _fetch_pool / _is_substantive_chunk。"""
    path = _repo_root() / "scripts" / "build_golden_test_set.py"
    spec = importlib.util.spec_from_file_location("_golden_chunk_pool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _truncate_passage(text: str, max_chars: int = 7500) -> str:
    t = text.replace("\x0c", "\n").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n\n…（下文已截断，仅用于生成问题）"


_SYSTEM_PROMPT = """你是 RAG 检索评测数据构造助手。用户会给你一段知识库中的文本片段。
你的任务是：根据片段内容，写出**一个**真实用户可能会搜索的问题。
输出要求严格：
- 只输出一行中文问句，不要编号、不要引号、不要前后缀说明。
- 问句应使「该片段」成为回答所依据的核心材料之一；避免空泛到任意文档都能答。
- 不要用「本段」「上文」「材料中」等指向文档结构的词。
- 避免连续照抄原文超过 12 个字符。"""


def _normalize_llm_question(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```[\w]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    for prefix in ("问句：", "问题：", "Q:", "Q：", "问："):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
    # 取首行
    line = s.splitlines()[0].strip() if s else ""
    line = line.strip("「」\"'“”")
    if len(line) > 200:
        line = line[:197] + "…"
    return line


def _fetch_chunk_text(conn: sqlite3.Connection, chunk_id: str, collection: str) -> str | None:
    row = conn.execute(
        "SELECT text FROM chunks WHERE id = ? AND collection_name = ?",
        (chunk_id, collection),
    ).fetchone()
    if not row:
        return None
    return row[0] or ""


def _regenerate_queries_only(args: argparse.Namespace) -> None:
    root = _repo_root()
    in_path = args.golden_set or args.out
    out_path = args.out

    with open(in_path, encoding="utf-8") as f:
        cases: List[dict] = json.load(f)

    llm = None
    if not args.dry_run:
        from src.core.settings import load_settings
        from src.libs.llm.llm_factory import LLMFactory

        settings = load_settings(args.config)
        llm = LLMFactory.create(settings)
        logger.info("已创建 LLM: provider=%s model=%s", llm.get_provider(), llm.get_model_name())

    conn = sqlite3.connect(str(args.db))
    n_total = len(cases)
    n_do = n_total if args.limit <= 0 else min(args.limit, n_total)

    try:
        for i in range(n_do):
            item = cases[i]
            gids = item.get("golden_chunk_ids") or []
            cid = gids[0] if gids else None
            coll = item.get("collection") or "cslearning"
            if not cid:
                logger.warning("[%d] 无 golden_chunk_ids，跳过", i)
                continue
            text = _fetch_chunk_text(conn, cid, coll)
            if text is None:
                logger.error("[%d] 库中无 chunk %s (%s)", i, cid, coll)
                if args.continue_on_error:
                    continue
                raise SystemExit(1)
            if args.dry_run:
                item["query"] = f"【DRY-RUN】{cid}"
                logger.info("[%d] dry-run 占位", i)
                continue
            assert llm is not None
            try:
                item["query"] = _generate_question_llm(llm, text)
                logger.info("[%d] OK: %s…", i, item["query"][:56])
            except Exception as e:
                logger.exception("[%d] LLM 失败 %s: %s", i, cid, e)
                if args.continue_on_error:
                    # 保留原 query，避免网络/SSL 等故障时整表被占位符覆盖
                    logger.warning("[%d] 已保留原 query，未改写", i)
                else:
                    raise SystemExit(f"生成失败: {cid} — {e}") from e
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        conn.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"已更新 {n_do}/{n_total} 条 query，写入 {out_path}")


def _generate_question_llm(llm: Any, passage: str) -> str:
    user_content = (
        "请根据下面文本，生成一个符合要求的检索问句（仅一行）。\n\n"
        f"{_truncate_passage(passage)}\n"
    )
    reply = llm.chat(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    q = _normalize_llm_question(reply)
    if not q or len(q) < 6:
        raise ValueError(f"模型返回过短或为空: {reply!r}")
    return q


def main() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(
        description="用 LLM 阅读每个 chunk 后生成 query，写入 golden_test_set.json",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=root / "data" / "db" / "rag.sqlite",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "tests" / "fixtures" / "golden_test_set.json",
    )
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-n", type=int, default=5)
    parser.add_argument("--cslearning-n", type=int, default=45)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用大模型，问句使用占位符（仅验证抽样与 JSON 结构）",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每次 API 调用后的休眠秒数，用于限流",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单条生成失败时写入占位问句并继续",
    )
    parser.add_argument(
        "--regenerate-queries-only",
        action="store_true",
        help="读取现有黄金集，仅按库中 chunk 用大模型重写 query，其余字段不变",
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=None,
        help="与 --regenerate-queries-only 联用：输入黄金集路径（默认与 --out 相同）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="与 --regenerate-queries-only 联用：只处理前 N 条（0 表示全部）",
    )
    args = parser.parse_args()

    if args.regenerate_queries_only:
        _regenerate_queries_only(args)
        return

    random.seed(args.seed)
    bgt = _load_chunk_pool_module()

    conn = sqlite3.connect(str(args.db))
    try:
        report_pool = bgt._fetch_pool(conn, "report", min_len=80)
        cs_pool = bgt._fetch_pool(conn, "cslearning", min_len=160)
    finally:
        conn.close()

    if len(report_pool) < args.report_n:
        raise SystemExit(f"report 可用 chunk 不足: {len(report_pool)} < {args.report_n}")
    if len(cs_pool) < args.cslearning_n:
        raise SystemExit(f"cslearning 可用 chunk 不足: {len(cs_pool)} < {args.cslearning_n}")

    report_pick = random.sample(report_pool, args.report_n)
    cs_pick = random.sample(cs_pool, args.cslearning_n)

    llm = None
    if not args.dry_run:
        from src.core.settings import load_settings
        from src.libs.llm.llm_factory import LLMFactory

        settings = load_settings(args.config)
        llm = LLMFactory.create(settings)
        logger.info("已创建 LLM: provider=%s model=%s", llm.get_provider(), llm.get_model_name())

    cases: List[dict] = []

    def one_case(
        i: int,
        cid: str,
        text: str,
        collection: str,
        label: str,
    ) -> dict:
        if args.dry_run:
            query = f"【DRY-RUN】请概述与 chunk {cid} 相关的主要内容"
        else:
            assert llm is not None
            try:
                query = _generate_question_llm(llm, text)
                logger.info("[%s] OK: %s…", label, query[:48])
            except Exception as e:
                logger.exception("[%s] LLM 失败 %s: %s", label, cid, e)
                if args.continue_on_error:
                    query = f"【生成失败】{cid}"
                else:
                    raise SystemExit(f"生成失败: {label} {cid} — {e}") from e
            if args.sleep > 0:
                time.sleep(args.sleep)
        return {
            "query": query,
            "golden_chunk_ids": [cid],
            "collection": collection,
            "description": f"{label} #{i + 1} ({cid})",
            "expected_content_checks": {"min_citations": 1},
        }

    for i, (cid, text) in enumerate(report_pick):
        cases.append(one_case(i, cid, text, "report", "report"))

    for i, (cid, text) in enumerate(cs_pick):
        cases.append(one_case(i, cid, text, "cslearning", "cslearning"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
