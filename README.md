# RAG_Qdrant_ZhiPu
A RAG-based question-answering system combining Zhipu's GLM-4 and Qdrant. It processes local documents, retrieves relevant fragments, and generates context-aware answers—ideal for knowledge bases and document queries.基于 RAG 的智能问答系统，结合智谱 GLM-4 与 Qdrant，处理本地文档、检索相关片段并生成上下文回答，适用于知识库、文档查询等场景。


# RAG 智能问答系统

基于检索增强生成（RAG）技术的问答系统，结合智谱AI大语言模型与Qdrant向量数据库，实现对本地文档的智能查询与精准回答。

## 项目概述

本项目通过以下流程实现智能问答：
1. 读取本地文档（支持PDF、Markdown、Word、TXT等格式）
2. 采用多种文本分割策略处理文档
3. 将文档向量存储至Qdrant向量数据库
4. 基于用户查询检索相关文档片段
5. 结合智谱GLM-4模型生成基于上下文的回答

## 环境要求

- Python 3.10
- Docker（用于运行Qdrant）
- Qdrant向量数据库

## 依赖安装

# 安装依赖包（根据实际导入库补充）
pip install langchain langchain-community llama-index qdrant-client python-dotenv zhipuai

# 配置说明
## 复制.env文件模板并配置：
ZHIPUAI_API_KEY="你的智谱API密钥"
collection_name="document_embeddings"
model='sentence_bert'
## 确保 Qdrant 服务已启动：
# 启动Docker和Qdrant（参考a.txt）
docker start qdrant
# 使用步骤
导入文档：将需要检索的文档放入./file目录，运行以下命令生成向量并存储：
python main1.py

测试检索（可选）：
python main2.py

智能问答：运行主程序并输入问题：
python main.py

# 文件结构
RAG/
├── file/               # 存放待处理的文档
├── main.py             # 问答主程序
├── main1.py            # 文档向量生成与存储
├── main2.py            # 检索测试程序
├── read_reg.py         # 文档读取与分割工具
├── qdrant_emded_store.py  # Qdrant向量存储操作
├── .env                # 环境变量配置
├── .gitignore          # Git忽略文件
└── a.txt               # 运行说明
# 注意事项
请确保智谱 API 密钥有效
处理大文档时可能需要调整分割参数（chunk_size、chunk_overlap）
首次运行需等待文档向量生成与存储完成

