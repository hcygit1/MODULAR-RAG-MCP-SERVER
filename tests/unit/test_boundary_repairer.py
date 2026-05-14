"""BoundaryRepairer unit tests."""

from src.core.settings import IngestionConfig
from src.ingestion.models import Chunk
from src.ingestion.transform.boundary_repairer import BoundaryRepairer


def _config() -> IngestionConfig:
    return IngestionConfig(
        chunk_size=512,
        chunk_overlap=50,
        enable_llm_refinement=False,
        enable_metadata_enrichment=False,
        enable_image_captioning=False,
        batch_size=16,
        enable_boundary_repair=True,
        boundary_repair_mode="rule",
    )


def test_boundary_repairer_repairs_normal_chunk() -> None:
    repairer = BoundaryRepairer(_config())
    chunks = [
        Chunk(id="c1", text="前文说明。"),
        Chunk(id="c2", text="这个普通块没有句号"),
        Chunk(id="c3", text="后续补充。"),
    ]

    repaired = repairer.repair(chunks)

    assert repaired[1].metadata["boundary_repair"] == "rule"
    assert repaired[1].text.endswith("后续补充。")


def test_boundary_repairer_skips_table_chunk_by_block_type() -> None:
    repairer = BoundaryRepairer(_config())
    chunks = [
        Chunk(id="c1", text="前文说明。"),
        Chunk(
            id="c2",
            text="| 指标 | 数值 |\n|---|---|\n| A | 1 |",
            metadata={"block_type": "table"},
        ),
        Chunk(id="c3", text="后续补充。"),
    ]

    repaired = repairer.repair(chunks)

    assert repaired[1].text == chunks[1].text
    assert "boundary_repair" not in repaired[1].metadata


def test_boundary_repairer_skips_table_chunk_by_structural_metadata() -> None:
    repairer = BoundaryRepairer(_config())
    chunks = [
        Chunk(id="c1", text="前文说明。"),
        Chunk(
            id="c2",
            text="| 指标 | 数值 |\n|---|---|\n| A | 1 |",
            metadata={"raw_table_html": "<table></table>", "row_range": [1, 10]},
        ),
        Chunk(id="c3", text="后续补充。"),
    ]

    repaired = repairer.repair(chunks)

    assert repaired[1].text == chunks[1].text
    assert "boundary_repair" not in repaired[1].metadata
