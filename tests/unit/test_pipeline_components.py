from unittest.mock import ANY, Mock, call

from modular_rag_mcp_server.core.trace.trace_context import TraceContext
from modular_rag_mcp_server.ingestion.document_chunk_builder import DocumentChunkBuilder
from modular_rag_mcp_server.ingestion.embedding.encoding_runner import EncodingRunner
from modular_rag_mcp_server.ingestion.models import Chunk, Document
from modular_rag_mcp_server.ingestion.pipeline import IngestionPipeline
from modular_rag_mcp_server.ingestion.storage.storage_coordinator import StorageCoordinator
from modular_rag_mcp_server.ingestion.transform.transform_runner import TransformRunner


def test_document_chunk_builder_preserves_structure_and_crops_image_assets():
    splitter = Mock()
    splitter.split_with_metadata.return_value = [
        ("alpha [IMAGE:img_a]", {"parent_id": "section_1"}),
        ("beta [IMAGE:img_b]", {"heading_text": "Second"}),
    ]
    builder = DocumentChunkBuilder(splitter)
    document = Document(
        id="doc_1",
        text="source text",
        metadata={
            "source_path": "/tmp/source.pdf",
            "image_data": {
                "img_a": b"image-a",
                "img_b": b"image-b",
                "unused": b"unused",
            },
            "images": [
                {"image_id": "img_a", "page": 1},
                {"image_id": "img_b", "page": 2},
                {"image_id": "unused", "page": 3},
            ],
        },
    )
    trace = object()

    chunks = builder.build(document, trace=trace)

    splitter.split_with_metadata.assert_called_once_with(
        document.text,
        doc_id=document.id,
        trace=trace,
    )
    assert [chunk.id for chunk in chunks] == ["doc_1_chunk_0", "doc_1_chunk_1"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, len("alpha [IMAGE:img_a]")),
        (
            len("alpha [IMAGE:img_a]"),
            len("alpha [IMAGE:img_a]") + len("beta [IMAGE:img_b]"),
        ),
    ]
    assert chunks[0].metadata["parent_id"] == "section_1"
    assert chunks[1].metadata["heading_text"] == "Second"
    assert chunks[0].metadata["image_refs"] == ["img_a"]
    assert chunks[0].metadata["image_data"] == {"img_a": b"image-a"}
    assert chunks[0].metadata["image_metadata"] == [
        {"image_id": "img_a", "page": 1}
    ]
    assert chunks[1].metadata["image_refs"] == ["img_b"]
    assert chunks[1].metadata["image_data"] == {"img_b": b"image-b"}
    assert chunks[1].metadata["image_metadata"] == [
        {"image_id": "img_b", "page": 2}
    ]
    assert chunks[0].metadata["source_path"] == "/tmp/source.pdf"
    assert "images" not in chunks[0].metadata
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source_doc_id"] == "doc_1"
    assert chunks[0].metadata["total_chunks"] == 2


def test_transform_runner_applies_transforms_in_fixed_order():
    events = []

    def make_transform(name):
        transform = Mock()

        def apply(chunk, trace=None):
            events.append((name, chunk.id, trace))
            return chunk

        transform.transform.side_effect = apply
        return transform

    runner = TransformRunner(
        chunk_refiner=make_transform("refine"),
        metadata_enricher=make_transform("enrich"),
        image_captioner=make_transform("caption"),
    )
    chunks = [
        Chunk(id="chunk_1", text="one"),
        Chunk(id="chunk_2", text="two"),
    ]
    trace = object()

    result = runner.run(chunks, trace=trace)

    assert result == chunks
    assert events == [
        ("refine", "chunk_1", trace),
        ("enrich", "chunk_1", trace),
        ("caption", "chunk_1", trace),
        ("refine", "chunk_2", trace),
        ("enrich", "chunk_2", trace),
        ("caption", "chunk_2", trace),
    ]


def test_transform_runner_keeps_transformed_chunks_when_boundary_repair_fails(
    caplog,
):
    repairer = Mock()
    repairer.repair.side_effect = RuntimeError("repair unavailable")
    runner = TransformRunner(
        chunk_refiner=Mock(),
        metadata_enricher=Mock(),
        image_captioner=Mock(),
        boundary_repairer=repairer,
    )
    chunks = [Chunk(id="chunk_1", text="already transformed")]
    trace = object()

    result = runner.repair_boundaries(chunks, trace=trace)

    assert result is chunks
    repairer.repair.assert_called_once_with(chunks, trace=trace)
    assert "块边界修复失败，跳过: repair unavailable" in caplog.text


def test_transform_runner_does_not_retry_when_stage_context_suppresses_error():
    class SuppressingStage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return True

    repairer = Mock()
    repairer.repair.side_effect = RuntimeError("repair unavailable")
    runner = TransformRunner(
        chunk_refiner=Mock(),
        metadata_enricher=Mock(),
        image_captioner=Mock(),
        boundary_repairer=repairer,
    )
    chunks = [Chunk(id="chunk_1", text="already transformed")]

    result = runner.repair_boundaries(
        chunks,
        stage_context=SuppressingStage(),
    )

    assert result is chunks
    repairer.repair.assert_called_once_with(chunks, trace=None)


def test_encoding_runner_delegates_arguments_to_batch_processor():
    batch_processor = Mock()
    batch_processor.process.return_value = ([[0.1]], [{"term": 1.0}])
    runner = EncodingRunner(batch_processor)
    chunks = [Chunk(id="chunk_1", text="content")]
    trace = object()

    result = runner.encode(chunks, trace=trace)

    assert result == ([[0.1]], [{"term": 1.0}])
    batch_processor.process.assert_called_once_with(chunks, trace=trace)


def test_storage_coordinator_replaces_chunks_for_a_document():
    vector_upserter = Mock()
    coordinator = StorageCoordinator(vector_upserter)
    chunks = [
        Chunk(
            id="doc_1_chunk_0",
            text="content",
            metadata={"source_doc_id": "doc_1"},
        )
    ]
    dense_vectors = [[0.1]]
    sparse_vectors = [{"term": 1.0}]
    trace = object()

    coordinator.store(
        chunks,
        dense_vectors,
        sparse_vectors,
        "collection",
        trace=trace,
    )

    vector_upserter.replace_document_chunks.assert_called_once_with(
        source_doc_id="doc_1",
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        trace=trace,
        collection_name="collection",
    )
    vector_upserter.upsert_chunks.assert_not_called()


def test_storage_coordinator_falls_back_when_document_replace_is_unsupported(
    caplog,
):
    vector_store = Mock()
    vector_store.get_backend.return_value = "legacy"
    vector_upserter = Mock()
    vector_upserter.replace_document_chunks.side_effect = NotImplementedError
    vector_upserter.get_vector_store.return_value = vector_store
    coordinator = StorageCoordinator(vector_upserter)
    chunks = [
        Chunk(
            id="doc_1_chunk_0",
            text="content",
            metadata={"source_doc_id": "doc_1"},
        )
    ]
    dense_vectors = [[0.1]]
    sparse_vectors = [{"term": 1.0}]
    trace = object()

    coordinator.store(
        chunks,
        dense_vectors,
        sparse_vectors,
        "collection",
        trace=trace,
    )

    assert vector_upserter.method_calls == [
        call.replace_document_chunks(
            source_doc_id="doc_1",
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            trace=trace,
            collection_name="collection",
        ),
        call.get_vector_store(),
        call.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            trace=trace,
            collection_name="collection",
        ),
    ]
    assert (
        "当前 VectorStore 不支持按 source_doc_id 清理，回退到普通 upsert。"
        "backend=legacy"
    ) in caplog.text


def test_pipeline_store_results_uses_current_compatibility_vector_upserter():
    old_upserter = Mock()
    current_upserter = Mock()
    pipeline = object.__new__(IngestionPipeline)
    coordinator = StorageCoordinator(old_upserter)
    pipeline._storage_coordinator = coordinator
    pipeline._vector_upserter = current_upserter
    chunks = [Chunk(id="chunk_1", text="content")]
    dense_vectors = [[0.1]]
    sparse_vectors = [{"term": 1.0}]
    trace = object()

    pipeline._store_results(
        chunks,
        dense_vectors,
        sparse_vectors,
        "collection",
        trace=trace,
    )

    current_upserter.upsert_chunks.assert_called_once_with(
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        trace=trace,
        collection_name="collection",
    )
    old_upserter.upsert_chunks.assert_not_called()
    assert pipeline._storage_coordinator is coordinator


def test_pipeline_process_chunks_uses_soft_boundary_repair_entry():
    pipeline = object.__new__(IngestionPipeline)
    chunks = [Chunk(id="chunk_1", text="content")]
    pipeline._transform_runner = Mock()
    pipeline._boundary_repairer = object()
    pipeline._transform_runner.run.return_value = chunks

    def repair_boundaries(
        current_chunks,
        trace=None,
        stage_context=None,
    ):
        with stage_context:
            return current_chunks

    pipeline._transform_runner.repair_boundaries.side_effect = (
        repair_boundaries
    )
    pipeline._encoding_runner = Mock()
    pipeline._encoding_runner.encode.return_value = (
        [[0.1]],
        [{"term": 1.0}],
    )
    pipeline._storage_coordinator = StorageCoordinator(Mock())
    trace = TraceContext(operation="ingestion")

    result = pipeline.process_chunks(chunks, "collection", trace=trace)

    assert result == 1
    pipeline._transform_runner.repair_boundaries.assert_called_once_with(
        chunks,
        trace=trace,
        stage_context=ANY,
    )
    assert "boundary_repair" in [
        stage["name"] for stage in trace.finish()["stages"]
    ]


def test_pipeline_soft_boundary_repair_preserves_trace_error_metadata():
    identity_transform = Mock()
    identity_transform.transform.side_effect = (
        lambda chunk, trace=None: chunk
    )
    repairer = Mock()
    repairer.repair.side_effect = RuntimeError("repair unavailable")
    pipeline = object.__new__(IngestionPipeline)
    pipeline._transform_runner = TransformRunner(
        chunk_refiner=identity_transform,
        metadata_enricher=identity_transform,
        image_captioner=identity_transform,
        boundary_repairer=repairer,
    )
    pipeline._boundary_repairer = repairer
    pipeline._encoding_runner = Mock()
    pipeline._encoding_runner.encode.return_value = (
        [[0.1]],
        [{"term": 1.0}],
    )
    pipeline._storage_coordinator = StorageCoordinator(Mock())
    trace = TraceContext(operation="ingestion")

    result = pipeline.process_chunks(
        [Chunk(id="chunk_1", text="content")],
        "collection",
        trace=trace,
    )

    boundary_stage = next(
        stage
        for stage in trace.finish()["stages"]
        if stage["name"] == "boundary_repair"
    )
    assert result == 1
    assert boundary_stage["metadata"]["error"] == "repair unavailable"


def test_pipeline_compatibility_dependencies_share_component_state():
    old_splitter = Mock()
    old_refiner = Mock()
    old_enricher = Mock()
    old_captioner = Mock()
    old_repairer = Mock()
    old_batch_processor = Mock()
    old_upserter = Mock()
    pipeline = object.__new__(IngestionPipeline)
    pipeline._chunk_builder = DocumentChunkBuilder(old_splitter)
    pipeline._transform_runner = TransformRunner(
        chunk_refiner=old_refiner,
        metadata_enricher=old_enricher,
        image_captioner=old_captioner,
        boundary_repairer=old_repairer,
    )
    pipeline._encoding_runner = EncodingRunner(old_batch_processor)
    pipeline._storage_coordinator = StorageCoordinator(old_upserter)

    new_splitter = Mock()
    new_refiner = Mock()
    new_enricher = Mock()
    new_captioner = Mock()
    new_repairer = Mock()
    new_batch_processor = Mock()
    new_upserter = Mock()
    pipeline._splitter = new_splitter
    pipeline._chunk_refiner = new_refiner
    pipeline._metadata_enricher = new_enricher
    pipeline._image_captioner = new_captioner
    pipeline._boundary_repairer = new_repairer
    pipeline._batch_processor = new_batch_processor
    pipeline._vector_upserter = new_upserter

    assert pipeline._chunk_builder.splitter is new_splitter
    assert pipeline._transform_runner.chunk_refiner is new_refiner
    assert pipeline._transform_runner.metadata_enricher is new_enricher
    assert pipeline._transform_runner.image_captioner is new_captioner
    assert pipeline._transform_runner.boundary_repairer is new_repairer
    assert pipeline._encoding_runner.batch_processor is new_batch_processor
    assert pipeline._storage_coordinator.vector_upserter is new_upserter
