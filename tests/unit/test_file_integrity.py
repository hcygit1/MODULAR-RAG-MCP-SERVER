"""
文件完整性检查单元测试

验证 SHA256 计算、跳过判断和成功标记功能。
"""
import tempfile
import os
import pytest
from pathlib import Path

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    compute_sha256,
    should_skip,
    mark_success
)


class TestFileIntegrityChecker:
    """FileIntegrityChecker 类测试"""
    
    def test_compute_sha256_success(self):
        """测试成功计算文件 SHA256"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            checker = FileIntegrityChecker()
            hash_value = checker.compute_sha256(temp_path)
            
            # 验证哈希值格式（64 个十六进制字符）
            assert len(hash_value) == 64
            assert all(c in '0123456789abcdef' for c in hash_value)
            
            # 验证同一文件多次计算得到相同哈希
            hash_value2 = checker.compute_sha256(temp_path)
            assert hash_value == hash_value2
        finally:
            os.unlink(temp_path)
    
    def test_compute_sha256_file_not_found(self):
        """测试文件不存在时抛出错误"""
        checker = FileIntegrityChecker()
        
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            checker.compute_sha256("/nonexistent/file.txt")
    
    def test_compute_sha256_different_files_different_hash(self):
        """测试不同文件产生不同哈希"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f1:
            f1.write("content 1")
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
            f2.write("content 2")
            temp_path2 = f2.name
        
        try:
            checker = FileIntegrityChecker()
            hash1 = checker.compute_sha256(temp_path1)
            hash2 = checker.compute_sha256(temp_path2)
            
            assert hash1 != hash2
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)
    
    def test_should_skip_returns_false_for_new_file(self):
        """测试新文件不应跳过"""
        checker = FileIntegrityChecker()
        test_hash = "test_hash_value_12345"
        
        assert checker.should_skip(test_hash) is False
    
    def test_should_skip_returns_true_after_mark_success(self):
        """测试标记成功后应跳过"""
        checker = FileIntegrityChecker()
        test_hash = "test_hash_value_67890"
        
        # 初始状态不应跳过
        assert checker.should_skip(test_hash) is False
        
        # 标记成功
        checker.mark_success(test_hash)
        
        # 现在应该跳过
        assert checker.should_skip(test_hash) is True
    
    def test_mark_success_persists(self):
        """测试标记成功状态持久化"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = Path(f.name)
        
        try:
            # 创建第一个检查器实例
            checker1 = FileIntegrityChecker(db_path=db_path)
            test_hash = "test_hash_persist"
            
            checker1.mark_success(test_hash)
            assert checker1.should_skip(test_hash) is True
            
            # 创建第二个检查器实例（使用相同数据库）
            checker2 = FileIntegrityChecker(db_path=db_path)
            
            # 应该仍然能检测到已处理
            assert checker2.should_skip(test_hash) is True
        finally:
            if db_path.exists():
                os.unlink(db_path)
    
    def test_clear_history(self):
        """测试清除历史记录"""
        checker = FileIntegrityChecker()
        test_hash = "test_hash_clear"
        
        checker.mark_success(test_hash)
        assert checker.should_skip(test_hash) is True
        
        # 清除特定记录
        checker.clear_history(file_hash=test_hash)
        assert checker.should_skip(test_hash) is False
        
        # 清除所有记录
        checker.mark_success(test_hash)
        checker.clear_history()
        assert checker.should_skip(test_hash) is False
    
    def test_mark_failed(self):
        """测试标记失败（失败的文件不应跳过）"""
        checker = FileIntegrityChecker()
        test_hash = "test_hash_failed"
        
        checker.mark_failed(test_hash)
        # 失败的文件不应跳过（只有 success 状态才跳过）
        assert checker.should_skip(test_hash) is False


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_compute_sha256_function(self):
        """测试便捷函数 compute_sha256"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test convenience function")
            temp_path = f.name
        
        try:
            hash_value = compute_sha256(temp_path)
            assert len(hash_value) == 64
            assert all(c in '0123456789abcdef' for c in hash_value)
        finally:
            os.unlink(temp_path)
    
    def test_should_skip_function(self):
        """测试便捷函数 should_skip"""
        test_hash = "test_convenience_hash"
        
        # 初始状态不应跳过
        assert should_skip(test_hash) is False
        
        # 标记成功后应跳过
        mark_success(test_hash)
        assert should_skip(test_hash) is True
    
    def test_mark_success_function(self):
        """测试便捷函数 mark_success"""
        test_hash = "test_mark_success_hash"
        
        assert should_skip(test_hash) is False
        mark_success(test_hash)
        assert should_skip(test_hash) is True


class TestIntegration:
    """集成测试：完整流程"""
    
    def test_full_workflow(self):
        """测试完整工作流程：计算哈希 -> 检查 -> 标记 -> 再次检查"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("integration test content")
            temp_path = f.name
        
        try:
            checker = FileIntegrityChecker()
            
            # 步骤 1: 计算文件哈希
            file_hash = checker.compute_sha256(temp_path)
            assert len(file_hash) == 64
            
            # 步骤 2: 检查是否应跳过（新文件，不应跳过）
            assert checker.should_skip(file_hash) is False
            
            # 步骤 3: 标记处理成功
            checker.mark_success(file_hash)
            
            # 步骤 4: 再次检查（应跳过）
            assert checker.should_skip(file_hash) is True
            
            # 步骤 5: 修改文件内容，重新计算哈希
            with open(temp_path, 'w') as f:
                f.write("modified content")
            
            new_hash = checker.compute_sha256(temp_path)
            assert new_hash != file_hash
            
            # 新哈希不应跳过
            assert checker.should_skip(new_hash) is False
        finally:
            os.unlink(temp_path)
    
    def test_same_content_same_hash(self):
        """测试相同内容产生相同哈希"""
        content = "same content"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f1:
            f1.write(content)
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
            f2.write(content)
            temp_path2 = f2.name
        
        try:
            checker = FileIntegrityChecker()
            hash1 = checker.compute_sha256(temp_path1)
            hash2 = checker.compute_sha256(temp_path2)
            
            # 相同内容应产生相同哈希
            assert hash1 == hash2
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)
