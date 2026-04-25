"""MinerU block parser tests."""

from src.libs.loader.mineru_block_parser import to_chunks
from src.libs.loader.mineru_cloud_client import MinerURawResult


def test_mineru_content_list_table_keeps_html_and_repeats_header() -> None:
    html = """
    <table>
      <tr><th>Name</th><th>Score</th></tr>
      <tr><td>Alice</td><td>90</td></tr>
      <tr><td>Bob</td><td>80</td></tr>
      <tr><td>Carl</td><td>70</td></tr>
    </table>
    """
    raw = MinerURawResult(
        markdown_text="",
        source_path="/tmp/report.pdf",
        images=[],
        content_list=[
            {"type": "text", "text": "Intro", "page_idx": 0},
            {
                "type": "table",
                "table_body": html,
                "table_caption": ["Scores"],
                "page_idx": 1,
                "bbox": [0, 10, 100, 200],
            },
        ],
    )

    chunks = to_chunks(raw, doc_id="doc_test", chunk_size=128, max_table_rows_per_chunk=2)

    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("block_type") == "table"]
    assert len(table_chunks) == 2
    assert "Name | Score" in table_chunks[0].text
    assert "Alice" in table_chunks[0].text
    assert "Name | Score" in table_chunks[1].text
    assert "Carl" in table_chunks[1].text
    assert table_chunks[0].metadata["raw_table_html"] == html
    assert table_chunks[0].metadata["row_range"] == [1, 2]
    assert table_chunks[0].metadata["page_idx"] == 1


def test_mineru_content_list_image_block_attaches_image_assets() -> None:
    raw = MinerURawResult(
        markdown_text="",
        source_path="/tmp/report.pdf",
        images=[("images/fig1.png", 2, b"png")],
        content_list=[
            {
                "type": "image",
                "img_path": "images/fig1.png",
                "image_caption": ["Architecture"],
                "page_idx": 2,
            }
        ],
    )

    chunks = to_chunks(raw, doc_id="doc_img", chunk_size=128)

    assert len(chunks) == 1
    assert "[IMAGE:doc_img_page_2_img_0]" in chunks[0].text
    assert chunks[0].metadata["image_refs"] == ["doc_img_page_2_img_0"]
    assert chunks[0].metadata["image_data"]["doc_img_page_2_img_0"] == b"png"
    assert chunks[0].metadata["image_metadata"][0]["page_idx"] == 2
