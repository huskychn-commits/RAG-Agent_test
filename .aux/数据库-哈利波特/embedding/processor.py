import torch
import os
import logging
import shutil  # 新增shutil模块用于文件移动
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import yaml

# 获取基础目录
BASE_DIR = os.getenv("pkl_base_dir", os.path.join(os.path.dirname((os.path.dirname(__file__)))))
BASE_DIR = os.path.normpath(BASE_DIR).replace("\\", "/")
CONFIG_FILE = os.path.join(BASE_DIR, "embedding", "config.yaml").replace("\\", "/")

# 加载配置文件
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)
    # 处理配置中的路径斜杠
    CONFIG["input_dir"] = CONFIG.get("input_dir", "").replace("\\", "/")
    CONFIG["output_dir"] = CONFIG.get("output_dir", "").replace("\\", "/")

# 从环境变量覆盖设备配置
CONFIG['device'] = os.getenv('DEVICE', CONFIG['device'])

class EmbeddingProcessor:
    def __init__(self):
        self.logger = self._setup_logger()
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        self.model = AutoModel.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])
        self._create_output_dir()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _create_output_dir(self):
        os.makedirs(CONFIG["output_dir"].replace("\\", "/"), exist_ok=True)
        self.logger.info(f"Output directory created: {CONFIG['output_dir']}")
    
    def _read_batch(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
            
    def generate_embeddings(self):
        input_path = CONFIG["input_dir"]
        for txt_file in os.listdir(input_path):
            if txt_file.endswith(".txt"):
                file_path = os.path.join(input_path, txt_file).replace("\\", "/")
                self.logger.info(f"Processing file: {file_path}")
                
                texts = self._read_batch(file_path)
                # 用于存储每个batch的综合embedding
                batch_embeddings = []
                
                for i in range(0, len(texts), CONFIG["batch_size"]):
                    batch = texts[i:i+CONFIG["batch_size"]]
                    # 合并batch中的所有文本生成一个综合文本（使用换行符连接）
                    combined_text = "\n".join(batch)
                    inputs = self.tokenizer(
                        combined_text,
                        padding=True,
                        truncation=True,
                        max_length=CONFIG["max_length"],
                        return_tensors='pt'
                    ).to(CONFIG["device"])
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                        if CONFIG["normalize"]:
                            batch_embedding = batch_embedding / np.linalg.norm(batch_embedding, axis=1, keepdims=True)
                        # 存储单个batch的embedding
                        batch_embeddings.append(batch_embedding.tolist()[0])
                
                # 将每个batch的embedding转换为numpy数组并保存
                all_embeddings = np.array(batch_embeddings)
                self._save_embeddings(all_embeddings, txt_file)
        
        # 移动临时文件并清理
        self._move_files()
    
    def _save_embeddings(self, vectors, filename):
        # 创建Faiss索引
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # 保存索引文件（规范化路径）
        output_dir = CONFIG.get("temp_dir", CONFIG["output_dir"])
        # 使用normpath确保路径格式正确
        output_file = os.path.normpath(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.index"))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # 直接使用系统默认编码处理路径
        faiss.write_index(index, output_file)
        self.logger.info(f"Saved embeddings to {output_file}")
    
    def _move_files(self):
        """将临时目录中的文件移回输出目录并清理临时文件"""
        temp_dir = CONFIG["temp_dir"]
        output_dir = CONFIG["output_dir"]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 移动所有.index文件
        for filename in os.listdir(temp_dir):
            if filename.endswith(".index"):
                src_path = os.path.join(temp_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.move(src_path, dst_path)
                self.logger.info(f"Moved file to output directory: {dst_path}")
                
        # 删除空的临时目录
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
                self.logger.info(f"Removed temporary directory: {temp_dir}")
            except OSError:
                self.logger.warning(f"Failed to remove temporary directory: {temp_dir} (可能非空)")

if __name__ == "__main__":
    processor = EmbeddingProcessor()
    processor.generate_embeddings()
