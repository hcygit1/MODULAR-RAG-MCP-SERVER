"""
文件完整性检查模块

提供文件 SHA256 哈希计算和增量处理跳过机制。
使用 SQLite 存储文件处理历史，支持零成本的增量更新。
"""
import hashlib
import sqlite3
import os
from pathlib import Path
from typing import Optional


# 默认数据库路径（在 cache 目录下）
DEFAULT_DB_PATH = Path("cache/processing/file_integrity.db")
LEGACY_COLLECTION_NAME = "__legacy__"


class FileIntegrityChecker:
    """
    文件完整性检查器
    
    负责计算文件 SHA256 哈希，并维护处理历史记录。
    支持判断文件是否已处理，避免重复处理。
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化文件完整性检查器
        
        Args:
            db_path: SQLite 数据库路径（可选），默认使用 cache/processing/file_integrity.db
        """
        self._db_path = db_path or DEFAULT_DB_PATH
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self) -> None:
        """确保数据库目录存在"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self) -> None:
        """初始化数据库表结构，并迁移旧版单主键表。"""
        with sqlite3.connect(str(self._db_path)) as conn:
            columns = conn.execute("PRAGMA table_info(ingestion_history)").fetchall()
            if not columns:
                self._create_table(conn)
            elif "collection_name" not in {column[1] for column in columns}:
                self._migrate_legacy_table(conn)
            conn.commit()

    @staticmethod
    def _create_table(conn: sqlite3.Connection) -> None:
        """创建按文件哈希和集合联合去重的历史表。"""
        conn.execute(
            """
            CREATE TABLE ingestion_history (
                file_hash TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                status TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (file_hash, collection_name)
            )
            """
        )

    def _migrate_legacy_table(self, conn: sqlite3.Connection) -> None:
        """将旧版 file_hash 单主键表迁移到保留作用域。"""
        conn.execute("ALTER TABLE ingestion_history RENAME TO ingestion_history_legacy")
        self._create_table(conn)
        conn.execute(
            """
            INSERT INTO ingestion_history (
                file_hash,
                collection_name,
                status,
                processed_at
            )
            SELECT file_hash, ?, status, processed_at
            FROM ingestion_history_legacy
            """,
            (LEGACY_COLLECTION_NAME,),
        )
        conn.execute("DROP TABLE ingestion_history_legacy")

    @staticmethod
    def _normalize_collection_name(collection_name: Optional[str]) -> str:
        """旧调用未传集合时使用兼容作用域。"""
        normalized = str(collection_name or "").strip()
        return normalized or LEGACY_COLLECTION_NAME
    
    def compute_sha256(self, file_path: str) -> str:
        """
        计算文件的 SHA256 哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件的 SHA256 哈希值（十六进制字符串）
            
        Raises:
            FileNotFoundError: 当文件不存在时
            IOError: 当文件无法读取时
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                # 分块读取大文件，避免内存问题
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
        except IOError as e:
            raise IOError(f"无法读取文件 {file_path}: {str(e)}") from e
        
        return sha256_hash.hexdigest()
    
    def should_skip(
        self,
        file_hash: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        判断文件是否应该跳过处理
        
        检查文件哈希是否已存在于数据库中且状态为 'success'。
        
        Args:
            file_hash: 文件的 SHA256 哈希值
            collection_name: 目标集合名称；不传时使用旧版兼容作用域
            
        Returns:
            bool: 如果文件已成功处理则返回 True，否则返回 False
        """
        collection = self._normalize_collection_name(collection_name)
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute("""
                SELECT status FROM ingestion_history 
                WHERE file_hash = ? AND collection_name = ? AND status = 'success'
            """, (file_hash, collection))
            result = cursor.fetchone()
            return result is not None
    
    def mark_success(
        self,
        file_hash: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        标记文件处理成功
        
        将文件哈希和成功状态写入数据库。如果记录已存在，则更新状态。
        
        Args:
            file_hash: 文件的 SHA256 哈希值
            collection_name: 目标集合名称；不传时使用旧版兼容作用域
        """
        collection = self._normalize_collection_name(collection_name)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_history (
                    file_hash,
                    collection_name,
                    status,
                    processed_at
                )
                VALUES (?, ?, 'success', CURRENT_TIMESTAMP)
            """, (file_hash, collection))
            conn.commit()
    
    def mark_failed(
        self,
        file_hash: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        标记文件处理失败（可选功能，用于调试）
        
        Args:
            file_hash: 文件的 SHA256 哈希值
            collection_name: 目标集合名称；不传时使用旧版兼容作用域
        """
        collection = self._normalize_collection_name(collection_name)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_history (
                    file_hash,
                    collection_name,
                    status,
                    processed_at
                )
                VALUES (?, ?, 'failed', CURRENT_TIMESTAMP)
            """, (file_hash, collection))
            conn.commit()
    
    def clear_history(
        self,
        file_hash: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        清除处理历史（主要用于测试）
        
        Args:
            file_hash: 可选的文件哈希
            collection_name: 可选的集合名称；与 file_hash 同时提供时只清除指定记录
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            if file_hash and collection_name is not None:
                collection = self._normalize_collection_name(collection_name)
                conn.execute(
                    """
                    DELETE FROM ingestion_history
                    WHERE file_hash = ? AND collection_name = ?
                    """,
                    (file_hash, collection),
                )
            elif file_hash:
                conn.execute("DELETE FROM ingestion_history WHERE file_hash = ?", (file_hash,))
            elif collection_name is not None:
                collection = self._normalize_collection_name(collection_name)
                conn.execute(
                    "DELETE FROM ingestion_history WHERE collection_name = ?",
                    (collection,),
                )
            else:
                conn.execute("DELETE FROM ingestion_history")
            conn.commit()


# 全局单例实例（可选，用于便捷访问）
_default_checker: Optional[FileIntegrityChecker] = None


def get_default_checker() -> FileIntegrityChecker:
    """获取默认的文件完整性检查器实例"""
    global _default_checker
    if _default_checker is None:
        _default_checker = FileIntegrityChecker()
    return _default_checker


# 便捷函数（符合规范要求的接口）
def compute_sha256(file_path: str) -> str:
    """
    计算文件的 SHA256 哈希值（便捷函数）
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件的 SHA256 哈希值
    """
    return get_default_checker().compute_sha256(file_path)


def should_skip(
    file_hash: str,
    collection_name: Optional[str] = None,
) -> bool:
    """
    判断文件是否应该跳过处理（便捷函数）
    
    Args:
        file_hash: 文件的 SHA256 哈希值
        collection_name: 目标集合名称（可选）
        
    Returns:
        bool: 如果文件已成功处理则返回 True
    """
    return get_default_checker().should_skip(file_hash, collection_name)


def mark_success(
    file_hash: str,
    collection_name: Optional[str] = None,
) -> None:
    """
    标记文件处理成功（便捷函数）
    
    Args:
        file_hash: 文件的 SHA256 哈希值
        collection_name: 目标集合名称（可选）
    """
    get_default_checker().mark_success(file_hash, collection_name)
