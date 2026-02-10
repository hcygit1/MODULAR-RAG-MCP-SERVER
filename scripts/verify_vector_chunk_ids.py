# 保存为 scripts/verify_vector_chunk_ids.py 后运行: uv run python scripts/verify_vector_chunk_ids.py
from qdrant_client import QdrantClient

client = QdrantClient(path="./data/db/qdrant")
info = client.get_collection("knowledge_base")

print("Collection: knowledge_base")
print("Points count:", info.points_count)

points, _ = client.scroll(
    collection_name="knowledge_base",
    limit=50,
    with_payload=True,
)

# 1. chunk id 格式
import re
pattern = re.compile(r"^doc_[a-f0-9]+_chunk_\d+$")
original_ids = []
for p in points:
    oid = (p.payload or {}).get("_original_id")
    if oid:
        original_ids.append(oid)

format_ok = all(pattern.match(oid) for oid in original_ids)
print("\n[1] Chunk ID format (doc_xxx_chunk_N):", "OK" if format_ok else "FAIL")

# 2. content_hash
has_hash = all("content_hash" in (p.payload or {}) for p in points)
print("[2] content_hash in metadata:", "OK" if has_hash else "FAIL")

# 3. 与 BM25 对齐（本 PDF 应有 21 个 chunk）
expected_ids = {f"doc_0462322bf398ef63_chunk_{i}" for i in range(21)}
actual_ids = set(original_ids)
align_ok = actual_ids == expected_ids
print("[3] Vector IDs == BM25 IDs (21 chunks):", "OK" if align_ok else "FAIL")
if not align_ok:
    print("    Missing:", expected_ids - actual_ids)
    print("    Extra:", actual_ids - expected_ids)

# 4. 抽样展示
print("\n--- Sample (first 3) ---")
for i, p in enumerate(points[:3]):
    pl = p.payload or {}
    print(f"  {pl.get('_original_id')!r} | content_hash={'Yes' if 'content_hash' in pl else 'No'} | text={str(pl.get('text', ''))[:40]}...")