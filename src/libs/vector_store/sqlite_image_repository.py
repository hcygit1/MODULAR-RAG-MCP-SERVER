import json
import sqlite3
from typing import Any, Dict, List, Optional, cast


class SQLiteImageRepository:
    """读写 SQLite 图片数据，不管理事务。"""

    def upsert_images(
        self,
        conn: sqlite3.Connection,
        chunks: List[Any],
        collection_name: str,
    ) -> None:
        all_image_ids: set[str] = set()
        image_data_map: Dict[str, bytes] = {}
        image_meta_map: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {})
            refs = metadata.get("image_refs") or []
            for image_id in refs:
                all_image_ids.add(image_id)
                chunk_image_data = metadata.get("image_data") or {}
                if image_id in chunk_image_data and image_id not in image_data_map:
                    image_data_map[image_id] = chunk_image_data[image_id]
                chunk_image_meta = metadata.get("image_metadata") or []
                for item in chunk_image_meta:
                    if item.get("image_id") == image_id and image_id not in image_meta_map:
                        image_meta_map[image_id] = {
                            key: value
                            for key, value in item.items()
                            if not isinstance(value, bytes)
                        }
                        break

        for image_id in all_image_ids:
            image_bytes = image_data_map.get(image_id)
            if not image_bytes:
                continue
            metadata = image_meta_map.get(image_id, {})
            mime_type = metadata.get("mime_type", "image/png")
            metadata_safe = {
                key: value
                for key, value in metadata.items()
                if key != "mime_type" and not isinstance(value, bytes)
            }
            metadata_json = (
                json.dumps(metadata_safe, ensure_ascii=False) if metadata_safe else None
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO images (
                    id, collection_name, image_data, mime_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (image_id, collection_name, image_bytes, mime_type, metadata_json),
            )

    def get_image(
        self,
        conn: sqlite3.Connection,
        image_id: str,
        collection_name: str,
    ) -> Optional[tuple[bytes, str]]:
        row = conn.execute(
            "SELECT image_data, mime_type FROM images WHERE id = ? AND collection_name = ?",
            (image_id, collection_name),
        ).fetchone()
        if row:
            return (row[0], row[1] or "image/png")
        return None

    def delete_orphan_images(
        self,
        conn: sqlite3.Connection,
        deleted_chunk_ids: List[str],
        collection_name: str,
    ) -> None:
        deleted_refs: set[str] = set()
        ids_json = json.dumps(deleted_chunk_ids, ensure_ascii=False)
        rows = conn.execute(
            """
            SELECT metadata_json
            FROM chunks
            WHERE collection_name = ?
              AND id IN (SELECT value FROM json_each(?))
            """,
            (collection_name, ids_json),
        ).fetchall()
        for row in rows:
            metadata = self._load_metadata(row[0])
            deleted_refs.update(metadata.get("image_refs") or [])
        if not deleted_refs:
            return

        remaining = conn.execute(
            """
            SELECT metadata_json
            FROM chunks
            WHERE collection_name = ?
              AND id NOT IN (SELECT value FROM json_each(?))
            """,
            (collection_name, ids_json),
        ).fetchall()
        still_referenced: set[str] = set()
        for row in remaining:
            metadata = self._load_metadata(row[0])
            still_referenced.update(metadata.get("image_refs") or [])

        for image_id in deleted_refs - still_referenced:
            conn.execute(
                "DELETE FROM images WHERE id = ? AND collection_name = ?",
                (image_id, collection_name),
            )

    def collect_image_refs(
        self,
        conn: sqlite3.Connection,
        chunk_ids: List[str],
        collection_name: str,
    ) -> set[str]:
        if not chunk_ids:
            return set()
        ids_json = json.dumps(chunk_ids, ensure_ascii=False)
        rows = conn.execute(
            """
            SELECT metadata_json
            FROM chunks
            WHERE collection_name = ?
              AND id IN (SELECT value FROM json_each(?))
            """,
            (collection_name, ids_json),
        ).fetchall()
        refs: set[str] = set()
        for row in rows:
            metadata = self._load_metadata(row[0])
            refs.update(metadata.get("image_refs") or [])
        return refs

    def delete_unreferenced_images(
        self,
        conn: sqlite3.Connection,
        candidate_image_ids: set[str],
        collection_name: str,
    ) -> None:
        if not candidate_image_ids:
            return
        rows = conn.execute(
            "SELECT metadata_json FROM chunks WHERE collection_name = ?",
            (collection_name,),
        ).fetchall()
        referenced: set[str] = set()
        for row in rows:
            metadata = self._load_metadata(row[0])
            referenced.update(metadata.get("image_refs") or [])
        for image_id in candidate_image_ids - referenced:
            conn.execute(
                "DELETE FROM images WHERE id = ? AND collection_name = ?",
                (image_id, collection_name),
            )

    @staticmethod
    def _load_metadata(raw_metadata: Optional[str]) -> Dict[str, Any]:
        if not raw_metadata:
            return {}
        try:
            return cast(Dict[str, Any], json.loads(raw_metadata))
        except json.JSONDecodeError:
            return {}
