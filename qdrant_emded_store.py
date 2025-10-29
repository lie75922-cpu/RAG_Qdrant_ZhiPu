# 使用Qdrant向量数据库，对LlamaIndex分割后的节点进行向量化与存储
import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量
collection_name = os.getenv("collection_name", "document_embeddings")
model = os.getenv("model", "sentence_bert")  


from sentence_transformers import SentenceTransformer
def text_to_vector_sentence_bert(text: str) -> list[float]:
    '''
    将文本转为向量（Sentence-BERT）
    :param text: 输入文本
    :return: 向量列表
    '''
    # 加载轻量模型（384维向量）
    sentence_bert = SentenceTransformer("all-MiniLM-L6-v2")

    """将文本转为向量（Sentence-BERT）"""
    result= sentence_bert.encode(text).tolist()

    return result


import torch
from transformers import AutoModel, AutoTokenizer
def text_to_vector_qwen(text: str) -> list[float]:
    '''
    将文本转为向量（Qwen-VL-Embedding）
    :param text: 输入文本
    :return: 向量列表
    '''
    # 加载Qwen-VL-Embedding模型（768维向量）
    model = AutoModel.from_pretrained(
        "qwen/Qwen-VL-Embedding",
        trust_remote_code=True,
        device_map="auto"  # 自动分配设备（CPU/GPU）
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "qwen/Qwen-VL-Embedding",
        trust_remote_code=True
    )
    # 将文本转为向量（Qwen-VL-Embedding）
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    # 输出格式为 PyTorch 张量,包含 input_ids（文本分词后的索引）、attention_mask等关键信息的字典。

    with torch.no_grad():
        embeddings = model(**inputs, task_type="text_embedding").last_hidden_state.mean(dim=1)
        # 模型最后一层的隐藏状态,[batch_size, sequence_length, hidden_size],[1, 文本分词后的长度, 768]
        # mean(dim=1) 对序列长度维度取平均，得到文本的整体向量表示，[1, 768]
    return embeddings.cpu().numpy().flatten().tolist() # flatten()展平为一维列表[768]


from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import uuid
def store_vectors_in_qdrant(collection_name: str,model:str, split_docs=None):
    """
    将向量存储到Qdrant向量数据库中
    :param vectors: 向量列表
    :param collection_name: Qdrant集合名称
    """
    # 连接Qdrant服务器
    client = QdrantClient(host="localhost", port=6333)

    # 向量维度
    if model == "sentence_bert":
        size=384
    else:
        size=768

    # 创建集合（如果不存在）
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=size, #向量的维度
            distance=Distance.COSINE # 文本相似性常用余弦距离
        )
    )

    # 准备插入数据（每个文档对应一个点（Point））
    points = []
    for doc in split_docs:
        # 生成向量
        if model == "sentence_bert":
            vector = text_to_vector_sentence_bert(doc.text)
        else:
            vector = text_to_vector_qwen(doc.text)
        
        # 构建点（含唯一ID、向量、元数据）
        points.append({
            "id": str(uuid.uuid4()),  # 唯一标识
            "vector": vector,
            "payload": {
                "text": doc.text,  # 原始文本
                "source": doc.metadata.get("source", "unknown"),  # 来源文件
                "page": doc.metadata.get("page", -1)  # 页码（PDF文档）
            }
        })
    # 批量插入数据点
    client.upsert(
        collection_name=collection_name,
        points=points
    )

#测试查询
def search_similar(query: str, top_k: int = 3) -> list:
    # 连接Qdrant服务器
    client = QdrantClient(host="localhost", port=6333)


    """搜索与查询文本相似的文档"""
    # 生成查询向量
    if model == "sentence_bert":
        query_vec = text_to_vector_sentence_bert(query)
    else:
        query_vec = text_to_vector_qwen(query)
    
    # 搜索相似向量
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k  # 返回前3个最相似结果
    )
    
    # 整理结果（提取元数据和相似度分数）
    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "page": hit.payload["page"],
            "score": hit.score  # 相似度分数（0~1，越高越相似）
        }
        for hit in results
    ]


if __name__ == "__main__":
    # 测试：生成"向量数据库"的嵌入
    vec = text_to_vector_sentence_bert("向量数据库用于存储和搜索高维向量")
    print(f"向量维度：{len(vec)}")  # 输出384
