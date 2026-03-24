#!/usr/bin/env python3
"""
从 rag.sqlite 抽样 report / cslearning 的 chunk，生成黄金测试集 JSON。

问题（query）由「从块中提炼的短主题」+「问句模板」生成，不使用正文长句截取。

用法（仓库根目录）:
  python scripts/build_golden_test_set.py
  python scripts/build_golden_test_set.py --out tests/fixtures/golden_test_set.json

大模型按块生成问句（无规则模板）: scripts/build_golden_test_set_llm.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_substantive_chunk(text: str, min_len: int = 160) -> bool:
    t = text.strip()
    if len(t) < min_len:
        return False
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    long_lines = [ln for ln in lines if len(ln) >= 20]
    if len(long_lines) < 2:
        return False
    alpha_zh = sum(1 for c in t if "\u4e00" <= c <= "\u9fff" or c.isalpha())
    if alpha_zh < 25 and "http" not in t.lower():
        return False
    return True


def _clean_body(text: str) -> str:
    t = text.replace("\x0c", "\n")
    t = re.sub(r"\[\d+\]", "", t)
    return t


def _collapse_cjk_spaces(s: str) -> str:
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s.strip())


def _extract_topic(text: str, collection: str) -> str:
    """
    从块中提炼短主题词/短语（标题、书名号、英文术语等），供模板填空。
    不返回完整句子级摘录。
    """
    raw = _clean_body(text)

    if collection == "report" and re.search(r"中\s*北\s*大\s*学|毕业设计\s*\(", raw[:400]):
        return "氢能车辆租赁与系统实现"

    m = re.search(r"《([^》]{2,28})》", raw)
    if m:
        return m.group(1).strip()[:26]

    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#"):
            h = re.sub(r"^#+\s*", "", s)
            h = re.sub(r"^\d+(\.\d+)*\s*", "", h).strip()
            h = re.sub(r"^第[一二三四五六七八九十\d]+章\s*", "", h)
            if 4 <= len(h) <= 36:
                return h[:30]

    ens = re.findall(
        r"(?:^|[^\w])([A-Za-z][A-Za-z0-9_\-]*(?:\s+[A-Za-z][a-z]+){0,4})",
        raw,
    )
    stop_en = {
        "http", "https", "note", "year", "title", "with", "from", "that", "this",
        "which", "there", "their", "github", "com", "schrodingercatss", "Version",
    }
    candidates: List[str] = []
    for e in ens:
        e2 = e.strip()
        low = e2.lower()
        if len(e2) < 4 or low in stop_en:
            continue
        if re.match(r"^[A-Z0-9_\-]+$", e2) and len(e2) < 20:
            candidates.append(e2)
        elif " " in e2 and len(e2) >= 6:
            candidates.append(e2[:40])
        elif len(e2) >= 5:
            candidates.append(e2[:40])
    if candidates:
        candidates.sort(key=len, reverse=True)
        pick = candidates[0]
        return _paraphrase_english_topic(pick, raw)

    for pat in (
        r"关于([^，。；\n]{4,12})",
        r"针对([^，。；\n]{4,12})",
        r"(?:包括|涉及)([^，。；\n]{4,12})",
    ):
        m2 = re.search(pat, raw)
        if m2:
            g = _collapse_cjk_spaces(m2.group(1).strip())
            g = re.sub(r"^(的|了|是)\s*", "", g)
            if 4 <= len(g) <= 14:
                return g

    zh_words = re.findall(r"[\u4e00-\u9fff]{5,14}", raw[:800])
    junk = {
        "因此", "然而", "可以", "进行", "问题", "方面", "主要", "内容", "相关", "技术",
        "方法", "研究", "分析", "发展", "系统", "实现", "设计", "应用", "情况", "需要",
    }
    for w in zh_words:
        w = _collapse_cjk_spaces(w)
        if w not in junk and not re.match(r"^\d+$", w) and len(w) <= 14:
            return w[:20]

    if collection == "report":
        if "氢能" in raw:
            return "氢能车辆与租赁服务"
        return "氢能交通与能源应用"
    return "深度学习超参数与实验流程"


def _paraphrase_english_topic(topic: str, raw: str) -> str:
    """英文术语/标题转短标签，避免问句里贴长段原文。"""
    t = topic.strip()
    low = raw.lower()
    if re.match(r"^[A-Za-z0-9_\-\s,\./:]+$", t) or (
        sum(1 for c in t if c.isalpha()) > len(t) * 0.5 and len(t) > 12
    ):
        if "playbook" in low or ("tuning" in low and "hyper" in low):
            return "深度学习调参与实验搜索"
        if "batch" in low and "norm" in low:
            return "Batch Norm 与批量大小"
        if "quasi" in low and "random" in low:
            return "准随机搜索与黑盒优化"
        if "stochastic" in low and "gradient" in low:
            return "随机梯度与优化过程"
        if len(t) > 22:
            return "文中的英文术语与方法论"
        return t[:22] + ("…" if len(t) > 22 else "")
    return t


def _question_templates(collection: str) -> List[str]:
    if collection == "report":
        return [
            "请概括文中围绕「{t}」展开的核心观点或结论。",
            "材料里与「{t}」相关的内容，主要想解决或说明什么问题？",
            "如果要向他人介绍「{t}」，依据该文可以强调哪些事实或判断？",
            "文中讨论「{t}」时，提到了哪些挑战、现状或对策？",
            "「{t}」在该文档语境下具体指什么？有哪些配套背景信息？",
        ]
    return [
        "请说明文中关于「{t}」的建议或原则性结论。",
        "材料在「{t}」这一主题下，主要解释了哪些机制或注意事项？",
        "若实践里要处理「{t}」相关问题，文中给出了哪些可操作的信息？",
        "「{t}」与上下文中的其他概念是如何衔接或区分的？",
        "围绕「{t}」，文档里有哪些值得记住的要点或常见误区？",
    ]


def _collection_fallback_topic(collection: str) -> str:
    return "氢能交通与租赁业务" if collection == "report" else "超参数实验与调优流程"


def _sanitize_topic(topic: str, collection: str) -> str:
    """去掉明显从目录/列表标题抄来的冗长主题，换成短标签。"""
    t = topic.strip()
    if len(t) > 16:
        return _collection_fallback_topic(collection)
    if re.match(r"^（[一二三四五六七八九十\d]+）", t) or "运营方" in t or "系统管理员" in t:
        return "系统角色与业务流程" if collection == "report" else "实验分工与流程"
    if any(k in t for k in ("大量的用户", "拟采用的技术", "用户数据和支付")):
        return "数据安全与支付设计" if collection == "report" else "数据与实验记录"
    return t


def _stable_template_index(chunk_id: str, n_templates: int) -> int:
    h = int(hashlib.md5(chunk_id.encode("utf-8")).hexdigest(), 16)
    return h % n_templates


def generate_question(text: str, collection: str, chunk_id: str) -> str:
    topic = _sanitize_topic(_extract_topic(text, collection), collection)
    templates = _question_templates(collection)
    idx = _stable_template_index(chunk_id, len(templates))
    q = templates[idx].format(t=topic)
    if len(q) > 200:
        q = q[:197] + "…"
    return q


def _fetch_pool(
    conn: sqlite3.Connection,
    collection: str,
    min_len: int,
) -> List[Tuple[str, str]]:
    cur = conn.execute(
        """
        SELECT id, text FROM chunks
        WHERE collection_name = ?
        """,
        (collection,),
    )
    rows = [(r[0], r[1]) for r in cur.fetchall() if _is_substantive_chunk(r[1], min_len)]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build golden_test_set.json from SQLite chunks.")
    parser.add_argument(
        "--db",
        type=Path,
        default=_repo_root() / "data" / "db" / "rag.sqlite",
        help="Path to rag.sqlite",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "tests" / "fixtures" / "golden_test_set.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--report-n", type=int, default=5)
    parser.add_argument("--cslearning-n", type=int, default=45)
    args = parser.parse_args()

    random.seed(args.seed)

    conn = sqlite3.connect(str(args.db))
    try:
        report_pool = _fetch_pool(conn, "report", min_len=80)
        cs_pool = _fetch_pool(conn, "cslearning", min_len=160)
    finally:
        conn.close()

    if len(report_pool) < args.report_n:
        raise SystemExit(f"report 可用 chunk 不足: {len(report_pool)} < {args.report_n}")
    if len(cs_pool) < args.cslearning_n:
        raise SystemExit(f"cslearning 可用 chunk 不足: {len(cs_pool)} < {args.cslearning_n}")

    report_pick = random.sample(report_pool, args.report_n)
    cs_pick = random.sample(cs_pool, args.cslearning_n)

    cases: List[dict] = []

    for i, (cid, text) in enumerate(report_pick):
        cases.append(
            {
                "query": generate_question(text, "report", cid),
                "golden_chunk_ids": [cid],
                "collection": "report",
                "description": f"report #{i + 1} ({cid})",
                "expected_content_checks": {"min_citations": 1},
            }
        )

    for i, (cid, text) in enumerate(cs_pick):
        cases.append(
            {
                "query": generate_question(text, "cslearning", cid),
                "golden_chunk_ids": [cid],
                "collection": "cslearning",
                "description": f"cslearning #{i + 1} ({cid})",
                "expected_content_checks": {"min_citations": 1},
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
