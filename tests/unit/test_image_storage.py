"""
ImageStorage 契约测试

验证图片文件存储和索引功能：
- 保存后文件存在
- 查找 image_id 返回正确路径
"""
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from base64 import b64encode

from src.ingestion.storage.image_storage import ImageStorage


class TestImageStorageBasic:
    """ImageStorage 基础功能测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_image_storage_initialization(self):
        """测试 ImageStorage 可以初始化"""
        storage = ImageStorage()
        assert storage is not None
        assert storage.get_total_images() == 0
    
    def test_image_storage_initialization_with_custom_path(self):
        """测试使用自定义路径初始化"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = ImageStorage(base_path=temp_dir)
            assert storage._base_path == Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_image_basic(self, temp_dir):
        """测试基本的图片保存功能"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 创建测试图片数据（简单的 PNG 文件头）
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        
        # 保存图片
        file_path = storage.save_image(
            image_id="test_image_1",
            image_data=image_data,
            collection_name="test_collection",
            metadata={"mime_type": "image/png", "source_doc": "test.pdf", "page": 1}
        )
        
        # 验证文件已创建
        assert Path(file_path).exists()
        assert Path(file_path).is_file()
        
        # 验证文件内容
        with open(file_path, "rb") as f:
            saved_data = f.read()
        assert saved_data == image_data
    
    def test_save_image_file_exists(self, temp_dir):
        """测试保存后文件存在（验收标准）"""
        storage = ImageStorage(base_path=temp_dir)
        
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        file_path = storage.save_image(
            image_id="test_image_1",
            image_data=image_data,
            collection_name="test_collection"
        )
        
        # 验收标准：保存后文件存在
        assert Path(file_path).exists(), "保存后文件应该存在"
    
    def test_get_image_path(self, temp_dir):
        """测试根据 image_id 查找路径（验收标准）"""
        storage = ImageStorage(base_path=temp_dir)
        
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        saved_path = storage.save_image(
            image_id="test_image_1",
            image_data=image_data,
            collection_name="test_collection"
        )
        
        # 验收标准：查找 image_id 返回正确路径
        retrieved_path = storage.get_image_path("test_image_1")
        assert retrieved_path == saved_path, "查找 image_id 应该返回正确路径"
        assert retrieved_path is not None
    
    def test_get_image_path_nonexistent(self, temp_dir):
        """测试查找不存在的 image_id"""
        storage = ImageStorage(base_path=temp_dir)
        
        path = storage.get_image_path("nonexistent_image")
        assert path is None
    
    def test_get_image_info(self, temp_dir):
        """测试获取图片完整信息"""
        storage = ImageStorage(base_path=temp_dir)
        
        metadata = {
            "mime_type": "image/png",
            "source_doc": "test.pdf",
            "page": 1,
            "width": 800,
            "height": 600
        }
        
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        storage.save_image(
            image_id="test_image_1",
            image_data=image_data,
            collection_name="test_collection",
            metadata=metadata
        )
        
        info = storage.get_image_info("test_image_1")
        assert info is not None
        assert info["image_id"] == "test_image_1"
        assert info["mime_type"] == "image/png"
        assert info["source_doc"] == "test.pdf"
        assert info["page"] == 1
        assert info["width"] == 800
        assert info["height"] == 600
    
    def test_save_image_from_base64(self, temp_dir):
        """测试从 Base64 编码保存图片"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 创建测试图片数据
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        base64_data = b64encode(image_data).decode("utf-8")
        
        # 保存
        file_path = storage.save_image_from_base64(
            image_id="test_image_base64",
            base64_data=base64_data,
            collection_name="test_collection",
            metadata={"mime_type": "image/png"}
        )
        
        # 验证文件已创建
        assert Path(file_path).exists()
        
        # 验证文件内容
        with open(file_path, "rb") as f:
            saved_data = f.read()
        assert saved_data == image_data
    
    def test_save_image_from_base64_with_data_uri(self, temp_dir):
        """测试从带 data URI 前缀的 Base64 保存"""
        storage = ImageStorage(base_path=temp_dir)
        
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        base64_data = b64encode(image_data).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_data}"
        
        file_path = storage.save_image_from_base64(
            image_id="test_image_data_uri",
            base64_data=data_uri,
            collection_name="test_collection"
        )
        
        assert Path(file_path).exists()
        with open(file_path, "rb") as f:
            saved_data = f.read()
        assert saved_data == image_data


class TestImageStorageIndex:
    """ImageStorage 索引管理测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_save_index(self, temp_dir):
        """测试保存索引"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 保存几张图片
        for i in range(3):
            image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
            storage.save_image(
                image_id=f"image_{i}",
                image_data=image_data,
                collection_name="test_collection"
            )
        
        # 保存索引
        storage.save_index("test_collection")
        
        # 验证索引文件已创建
        index_file = Path(temp_dir) / "test_collection" / "index.json"
        assert index_file.exists()
        
        # 验证索引文件内容
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        
        assert index_data["collection_name"] == "test_collection"
        assert index_data["total_images"] == 3
        assert "images" in index_data
        assert len(index_data["images"]) == 3
    
    def test_load_index(self, temp_dir):
        """测试加载索引"""
        storage1 = ImageStorage(base_path=temp_dir)
        
        # 保存图片和索引
        image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
        storage1.save_image(
            image_id="test_image_1",
            image_data=image_data,
            collection_name="test_collection"
        )
        storage1.save_index("test_collection")
        
        # 创建新的 storage 并加载索引
        storage2 = ImageStorage(base_path=temp_dir)
        storage2.load_index("test_collection")
        
        # 验证索引已加载
        assert storage2.get_total_images() == 1
        path = storage2.get_image_path("test_image_1")
        assert path is not None
    
    def test_load_index_nonexistent(self, temp_dir):
        """测试加载不存在的索引"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 应该不会抛出异常，只是使用空索引
        storage.load_index("nonexistent_collection")
        assert storage.get_total_images() == 0
    
    def test_load_index_invalid_format(self, temp_dir):
        """测试加载格式错误的索引文件"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 创建格式错误的索引文件
        collection_path = Path(temp_dir) / "test_collection"
        collection_path.mkdir(parents=True, exist_ok=True)
        index_file = collection_path / "index.json"
        
        with open(index_file, "w", encoding="utf-8") as f:
            f.write("invalid json content")
        
        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="加载索引文件失败"):
            storage.load_index("test_collection")


class TestImageStorageEdgeCases:
    """ImageStorage 边界情况测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_save_image_empty_id(self, temp_dir):
        """测试空 image_id"""
        storage = ImageStorage(base_path=temp_dir)
        
        with pytest.raises(ValueError, match="image_id 不能为空"):
            storage.save_image("", b"data", "collection")
    
    def test_save_image_empty_data(self, temp_dir):
        """测试空图片数据"""
        storage = ImageStorage(base_path=temp_dir)
        
        with pytest.raises(ValueError, match="image_data 不能为空"):
            storage.save_image("test_id", b"", "collection")
    
    def test_save_image_empty_collection(self, temp_dir):
        """测试空集合名称"""
        storage = ImageStorage(base_path=temp_dir)
        
        with pytest.raises(ValueError, match="collection_name 不能为空"):
            storage.save_image("test_id", b"data", "")
    
    def test_save_image_invalid_base64(self, temp_dir):
        """测试无效的 Base64 数据"""
        storage = ImageStorage(base_path=temp_dir)
        
        with pytest.raises(ValueError, match="Base64 解码失败"):
            storage.save_image_from_base64("test_id", "invalid_base64!!!", "collection")
    
    def test_get_extension_from_mime_type(self, temp_dir):
        """测试根据 MIME 类型获取扩展名"""
        storage = ImageStorage(base_path=temp_dir)
        
        assert storage._get_extension_from_mime_type("image/png") == ".png"
        assert storage._get_extension_from_mime_type("image/jpeg") == ".jpg"
        assert storage._get_extension_from_mime_type("image/jpg") == ".jpg"
        assert storage._get_extension_from_mime_type("image/gif") == ".gif"
        assert storage._get_extension_from_mime_type("image/webp") == ".webp"
        assert storage._get_extension_from_mime_type("image/svg+xml") == ".svg"
        assert storage._get_extension_from_mime_type("unknown/type") == ".png"  # 默认
    
    def test_list_images(self, temp_dir):
        """测试列出所有图片 ID"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 保存多张图片
        for i in range(5):
            image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
            storage.save_image(
                image_id=f"image_{i}",
                image_data=image_data,
                collection_name="test_collection"
            )
        
        # 列出所有图片
        image_ids = storage.list_images()
        assert len(image_ids) == 5
        assert "image_0" in image_ids
        assert "image_4" in image_ids
    
    def test_list_images_by_collection(self, temp_dir):
        """测试按集合列出图片"""
        storage = ImageStorage(base_path=temp_dir)
        
        # 保存到不同集合
        for i in range(3):
            image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
            storage.save_image(
                image_id=f"image_{i}",
                image_data=image_data,
                collection_name="collection_1"
            )
        
        for i in range(2):
            image_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100
            storage.save_image(
                image_id=f"image_{i+3}",
                image_data=image_data,
                collection_name="collection_2"
            )
        
        # 按集合列出
        ids_1 = storage.list_images("collection_1")
        assert len(ids_1) == 3
        
        ids_2 = storage.list_images("collection_2")
        assert len(ids_2) == 2
    
    def test_save_index_empty_collection(self, temp_dir):
        """测试保存空集合的索引"""
        storage = ImageStorage(base_path=temp_dir)
        
        with pytest.raises(ValueError, match="collection_name 不能为空"):
            storage.save_index("")
    
    def test_load_index_empty_collection(self, temp_dir):
        """测试加载空集合名称的索引"""
        storage = ImageStorage(base_path=temp_dir)
        
        with pytest.raises(ValueError, match="collection_name 不能为空"):
            storage.load_index("")
