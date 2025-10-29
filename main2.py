# main2.py
# 测试向量检索

import read_reg
import qdrant_emded_store
from qdrant_client import QdrantClient


# 测试查询
query = "操作票"
similar_docs = qdrant_emded_store.search_similar(query)
for i, doc in enumerate(similar_docs, 1):
    print(f"\n第{i}个相似文档（分数：{doc['score']:.2f}）：")
    print(f"来源：{doc['source']}（第{doc['page']}页）")
    print(f"内容：{doc['text'][:200]}...")  # 打印前200字
