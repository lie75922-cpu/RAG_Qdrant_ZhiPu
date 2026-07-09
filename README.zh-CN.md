# RAGForgeX

RAGForgeX 是一个模块化 RAG 实验工具包，用于快速构建、切换和评估 Retrieval-Augmented Generation 流水线。它为文档解析、文本切分、向量模型、向量数据库、检索器、重排器、生成器和评估器提供统一接口，让用户通过 YAML 配置比较不同 RAG 方案，而不是反复重写粘合代码。

其他语言：

- [English](README.md)
- [한국어](README.ko.md)

## V2 新增内容

- 新增 Chroma 和 Milvus 向量存储适配器，并保留本地 fallback。
- 新增语义切分和父子切分，适合更复杂的索引结构。
- 新增 query rewrite 检索，用于轻量多查询实验。
- 新增 BGE reranker 和 DeepEval evaluator 适配器，依赖缺失时会优雅降级。
- 扩展配置、文档、测试和多语言 README。

## 快速开始

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
ragforgex run --config configs/faiss_dense.yaml --question "What is RAGForgeX?"
```

运行 V2 示例：

```bash
ragforgex run --config configs/chroma_semantic.yaml --question "What does the toolkit compare?"
```

## 核心能力

| 层级 | V2 支持 |
| --- | --- |
| Parser | 文本 fallback、Docling、LlamaIndex |
| Chunker | Recursive、Token、Semantic、Parent-child |
| Embedding | 本地确定性 fallback、SentenceTransformers、OpenAI-compatible |
| Vector Store | In-memory、Qdrant、FAISS、Chroma、Milvus |
| Retriever | Dense、BM25、Hybrid、RRF、Query rewrite |
| Reranker | No-op、BGE adapter |
| Generator | OpenAI-compatible、Zhipu、Ollama、fallback |
| Evaluation | Retrieval metrics、Ragas adapter、DeepEval adapter |

## 后续优化方案

- 增加可持久化索引和实验产物，便于复现实验结果。
- 增加 typed config validation，输出更清晰的配置错误。
- 支持异步文档入库、批量 embedding 和流式生成。
- 增加图谱抽取、Neo4j 存储和 GraphRAG 示例。
- 增加跨数据集、检索器和生成器的标准化评估报告。

