# RAGForgeX

RAGForgeX는 Retrieval-Augmented Generation 파이프라인을 빠르게 만들고, 교체하고, 평가하기 위한 모듈형 RAG 실험 도구입니다. 문서 파서, 청커, 임베딩 모델, 벡터 스토어, 리트리버, 리랭커, 생성기, 평가기에 공통 인터페이스를 제공하므로 사용자는 접착 코드를 반복해서 작성하지 않고 YAML 설정으로 다양한 RAG 설계를 비교할 수 있습니다.

다른 언어:

- [English](README.md)
- [中文](README.zh-CN.md)

## V2 새로운 기능

- Chroma 및 Milvus 벡터 스토어 어댑터와 로컬 fallback 동작을 추가했습니다.
- Semantic chunker와 parent-child chunker를 추가했습니다.
- 간단한 multi-query 실험을 위한 query rewrite retriever를 추가했습니다.
- BGE reranker와 DeepEval evaluator 어댑터를 추가했으며 선택 의존성이 없을 때도 안전하게 동작합니다.
- 설정, 문서, 테스트, 다국어 README를 확장했습니다.

## 빠른 시작

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
ragforgex run --config configs/faiss_dense.yaml --question "What is RAGForgeX?"
```

V2 예시 실행:

```bash
ragforgex run --config configs/chroma_semantic.yaml --question "What does the toolkit compare?"
```

## 주요 기능

| 계층 | V2 지원 |
| --- | --- |
| Parser | Text fallback, Docling, LlamaIndex |
| Chunker | Recursive, Token, Semantic, Parent-child |
| Embedding | Deterministic fallback, SentenceTransformers, OpenAI-compatible |
| Vector Store | In-memory, Qdrant, FAISS, Chroma, Milvus |
| Retriever | Dense, BM25, Hybrid, RRF, Query rewrite |
| Reranker | No-op, BGE adapter |
| Generator | OpenAI-compatible, Zhipu, Ollama, fallback |
| Evaluation | Retrieval metrics, Ragas adapter, DeepEval adapter |

## 향후 최적화 계획

- 재현 가능한 실험을 위해 인덱스와 결과물을 저장합니다.
- 타입 기반 설정 검증과 더 명확한 오류 메시지를 추가합니다.
- 비동기 수집, 배치 임베딩, 스트리밍 생성을 지원합니다.
- 그래프 추출, Neo4j 저장소, GraphRAG 예제를 추가합니다.
- 데이터셋, 리트리버, 생성기를 비교하는 표준 평가 리포트를 제공합니다.

