"""
HeadingSplitter 单元测试

覆盖：模式 A（多菜文件，有 ###）、模式 B（单菜文件，无 ###）、
边界情况（无 ##、只有 #、连续 ###）、split_text 兼容接口。
"""
import pytest

from src.core.settings import IngestionConfig
from src.libs.splitter.heading_splitter import HeadingSplitter


def _make_splitter(parent_level=2, split_level=3) -> HeadingSplitter:
    cfg = IngestionConfig(
        chunk_size=512, chunk_overlap=0,
        enable_llm_refinement=False, enable_metadata_enrichment=False,
        enable_image_captioning=False, batch_size=32,
        splitter_strategy="heading",
        heading_parent_level=parent_level,
        heading_split_level=split_level,
    )
    return HeadingSplitter(cfg)


# ---------------------------------------------------------------------------
# 模式 A：文档含 ### （多菜文件）
# ---------------------------------------------------------------------------

MULTI_DISH_MD = """# 川菜菜谱

## 宫保鸡丁

### 食材
鸡肉300g，花生50g

### 步骤
热锅翻炒

## 鱼香肉丝

### 食材
猪肉丝200g

### 步骤
调鱼香汁
"""


def test_mode_a_parent_ids():
    """同一道菜的子块共享 parent_id，不同道菜不同 parent_id。"""
    s = _make_splitter()
    results = s.split_with_metadata(MULTI_DISH_MD, doc_id="doc")
    by_parent: dict = {}
    for _, meta in results:
        pid = meta["parent_id"]
        by_parent.setdefault(pid, []).append(meta["heading_text"])

    # 宫保鸡丁的节都应在同一个 parent_id 下
    parent_headings = list(by_parent.values())
    dish1_chunks = [v for v in by_parent.values() if "宫保鸡丁" in v]
    dish2_chunks = [v for v in by_parent.values() if "鱼香肉丝" in v]
    assert len(dish1_chunks) == 1, "宫保鸡丁的子块应属于同一父"
    assert len(dish2_chunks) == 1, "鱼香肉丝的子块应属于同一父"
    # 两道菜的 parent_id 不同
    assert len(by_parent) >= 2


def test_mode_a_heading_text():
    """子块 heading_text 应正确提取。"""
    s = _make_splitter()
    results = s.split_with_metadata(MULTI_DISH_MD, doc_id="doc")
    headings = [meta["heading_text"] for _, meta in results]
    assert "食材" in headings
    assert "步骤" in headings


def test_mode_a_chunk_contains_content():
    """子块文本应包含标题行和内容。"""
    s = _make_splitter()
    results = s.split_with_metadata(MULTI_DISH_MD, doc_id="doc")
    food_chunks = [(t, m) for t, m in results if m["heading_text"] == "食材" and "鸡肉" in t]
    assert len(food_chunks) >= 1


# ---------------------------------------------------------------------------
# 模式 B：文档无 ### （单菜文件）
# ---------------------------------------------------------------------------

SINGLE_DISH_MD = """# 皮蛋豆腐的做法

皮蛋豆腐是一道简单菜。

## 必备原料和工具

- 皮蛋
- 内酯豆腐

## 操作

先把皮蛋剥壳，切四瓣。
"""


def test_mode_b_all_share_one_parent():
    """单菜文件：所有子块共享同一 parent_id。"""
    s = _make_splitter()
    results = s.split_with_metadata(SINGLE_DISH_MD, doc_id="皮蛋豆腐")
    parent_ids = {meta["parent_id"] for _, meta in results}
    assert len(parent_ids) == 1, f"单菜文件应只有一个 parent_id，得到：{parent_ids}"


def test_mode_b_parent_id_format():
    """模式 B 的 parent_id 应以 _section_0 结尾。"""
    s = _make_splitter()
    results = s.split_with_metadata(SINGLE_DISH_MD, doc_id="皮蛋豆腐")
    pid = results[0][1]["parent_id"]
    assert pid == "皮蛋豆腐_section_0"


def test_mode_b_splits_at_parent_level():
    """模式 B 应在 ## 处切块，每个 ## 节各一个子块（加开头段落）。"""
    s = _make_splitter()
    results = s.split_with_metadata(SINGLE_DISH_MD, doc_id="doc")
    headings = [meta["heading_text"] for _, meta in results]
    assert "必备原料和工具" in headings
    assert "操作" in headings


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------

def test_no_headings_returns_one_chunk():
    """无任何标题的纯文本：整段作为一个子块返回。"""
    s = _make_splitter()
    text = "这是一段普通文字，没有任何 Markdown 标题。"
    results = s.split_with_metadata(text, doc_id="doc")
    assert len(results) == 1
    assert results[0][0] == text


def test_only_top_level_heading():
    """只有 # 标题（比父级还高），不触发父分组，归入兜底。"""
    s = _make_splitter()
    text = "# 只有一级标题\n\n这是内容。"
    results = s.split_with_metadata(text, doc_id="doc")
    assert len(results) == 1


def test_consecutive_split_level_headings():
    """连续多个 ### 块（模式 A），每个都是独立子块，parent_id 相同。"""
    s = _make_splitter()
    text = "## 宫保鸡丁\n\n### 食材\nA\n\n### 步骤\nB\n\n### 小贴士\nC"
    results = s.split_with_metadata(text, doc_id="doc")
    parent_ids = {m["parent_id"] for _, m in results}
    assert len(parent_ids) == 1  # section_0 兜底 + section_1 宫保鸡丁，但 ### 都在 section_1


def test_empty_text_raises():
    """空文本应抛出 ValueError。"""
    s = _make_splitter()
    with pytest.raises(ValueError):
        s.split_with_metadata("", doc_id="doc")


# ---------------------------------------------------------------------------
# split_text 兼容接口
# ---------------------------------------------------------------------------

def test_split_text_returns_strings():
    """split_text 返回字符串列表，不包含 metadata。"""
    s = _make_splitter()
    chunks = s.split_text(SINGLE_DISH_MD)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) > 0


def test_split_with_metadata_count_matches_split_text():
    """split_with_metadata 与 split_text 的块数一致。"""
    s = _make_splitter()
    tuples = s.split_with_metadata(MULTI_DISH_MD, doc_id="doc")
    texts = s.split_text(MULTI_DISH_MD)
    assert len(tuples) == len(texts)
