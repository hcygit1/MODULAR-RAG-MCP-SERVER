"""
数据摄取 E2E 测试

验证 ingest.py 脚本的端到端功能：
- 命令行可运行并在 data/db 产生产物
- 重复运行在未变更时跳过
"""
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_ingest(e2e_runtime, *args):
    """在隔离目录中运行 ingest.py。"""
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingest.py"),
            *args,
            "--config",
            str(e2e_runtime["config_path"]),
        ],
        capture_output=True,
        text=True,
        cwd=e2e_runtime["workdir"],
        env=e2e_runtime["env"],
    )


class TestDataIngestionE2E:
    """数据摄取 E2E 测试"""
    
    def test_ingest_script_help(self):
        """测试脚本帮助信息"""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "--collection" in result.stdout
        assert "--path" in result.stdout
        assert "--force" in result.stdout
    
    def test_ingest_script_missing_args(self):
        """测试缺少必需参数时的错误处理"""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest.py")],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_ingest_script_with_sample_pdf(self, e2e_runtime, sample_pdf_path):
        """测试使用样例 PDF 运行 ingest 脚本"""
        result = _run_ingest(
            e2e_runtime,
            "--collection",
            "test_collection",
            "--path",
            str(sample_pdf_path),
        )
        
        assert result.returncode == 0, result.stderr
        with sqlite3.connect(e2e_runtime["sqlite_path"]) as conn:
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE collection_name = ?",
                ("test_collection",),
            ).fetchone()[0]
        assert chunk_count > 0
        
    def test_ingest_script_skip_unchanged_file(self, e2e_runtime, sample_pdf_path):
        """测试重复运行时会跳过未变更的文件"""
        result1 = _run_ingest(
            e2e_runtime,
            "--collection",
            "test_collection_skip",
            "--path",
            str(sample_pdf_path),
        )
        result2 = _run_ingest(
            e2e_runtime,
            "--collection",
            "test_collection_skip",
            "--path",
            str(sample_pdf_path),
        )
        
        assert result1.returncode == 0, result1.stderr
        assert result2.returncode == 0, result2.stderr
        assert "文件跳过" in result2.stderr

    def test_ingest_script_force_flag(self, e2e_runtime, sample_pdf_path):
        """测试 --force 标志"""
        result = _run_ingest(
            e2e_runtime,
            "--collection",
            "test_collection_force",
            "--path",
            str(sample_pdf_path),
            "--force",
        )
        
        assert result.returncode == 0, result.stderr
        assert "强制模式" in result.stderr
    
    def test_ingest_script_invalid_path(self):
        """测试无效路径的错误处理"""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "ingest.py"),
                "--collection",
                "test_collection",
                "--path",
                "nonexistent_file.pdf",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        
        assert result.returncode != 0
        assert "路径不存在" in result.stderr
    
    def test_ingest_script_directory_path(self, e2e_runtime):
        """测试目录路径处理"""
        test_dir = Path(e2e_runtime["workdir"]) / "test_docs"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        result = _run_ingest(
            e2e_runtime,
            "--collection",
            "test_collection_dir",
            "--path",
            str(test_dir),
        )
        
        assert result.returncode == 0
        assert "目录中未找到 PDF 文件" in result.stderr
