# main1.py
# 向量存储到Qdrant

import read_reg
import qdrant_emded_store

import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

collection_name = os.getenv("collection_name", "document_embeddings")
model = os.getenv("model", "sentence_bert")  


# 读取文件
docs = read_reg.read_file_llama("./file")

# 分割文档
split_docs = read_reg.split_documents_recursive(docs, chunk_size=1000, chunk_overlap=100)
print(f"分割后节点数: {len(split_docs)}")
print(split_docs[:2])  # 打印前两个分割后的节点

# 存储向量到Qdrant
qdrant_emded_store.store_vectors_in_qdrant(
    collection_name=collection_name, 
    model=model,
    split_docs=split_docs
)

print("向量存储到Qdrant完成。")