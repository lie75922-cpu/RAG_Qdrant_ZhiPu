# 读取文件
# 使用不同的分割器将文档分割为更小的节点
# 提供多种分割方法，适应不同类型的文本内容，根据需求选择合适的分割器

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter 


def read_file_llama(document_dir:str = "./file"):
    """
    使用LlamaIndex加载目录下的文件，支持多种格式（PDF、Markdown、Word等）
    :param document_dir: 文件目录路径
    :return: 加载的文档列表

    """
    # 加载目录下所有支持的文件（PDF、Markdown、Word等）
    loader = SimpleDirectoryReader(
        input_dir=document_dir,  # 目标目录
        required_exts=[".pdf", ".md", ".docx",".txt"],  # 限定格式（可选）
    )
    docs = loader.load_data()  # 列表，每个元素是LlamaIndex的Document对象
    return docs

def split_documents_token(documents, chunk_size:int=512, chunk_overlap:int=20):
    """
    使用TokenSplitter将文档分割为更小的节点，适合处理长文本
    :param documents: LlamaIndex的Document对象列表
    :return: 分割后的节点列表
    """
    # 1. 初始化TokenSplitter（自定义参数）
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,  # 每个块最多512 tokens
        chunk_overlap=chunk_overlap  # 块之间重叠20 tokens
    )

    # 2. 分割文档（将Document转为多个Node）
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
def split_documents_recursive(documents, chunk_size:int=1000, chunk_overlap:int=100):
    """
    使用LangchainNodeParser中的RecursiveCharacterTextSplitter将文档分割为更小的节点，适合处理中文文本
    :param documents: LlamaIndex的Document对象列表
    :return: 分割后的节点列表
    """
    # 初始化递归字符分割器（适合中文文本）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 每个块最多字符
        chunk_overlap=chunk_overlap,  # 重叠字符
        separators=["\n\n", "\n", "。", "，", " "]  # 中文优先按段落、换行、句号分割
    )
    parser = LangchainNodeParser(splitter)
    # 分割文档
    nodes = parser.get_nodes_from_documents(documents)
    return nodes


from llama_index.core.node_parser import MarkdownNodeParser
def split_documents_markdown(documents, chunk_size:int=512, chunk_overlap:int=20):
    """
    使用MarkdownSplitter将Markdown文档分割为更小的节点
    :param documents: LlamaIndex的Document对象列表
    :return: 分割后的节点列表
    """
    # 初始化Markdown分割器
    splitter = MarkdownNodeParser(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    nodes = splitter.get_nodes_from_documents(documents)  # 输入需是Markdown文档
    return nodes

from llama_index.core.node_parser import CodeSplitter
def split_documents_code(documents):
    """
    使用CodeSplitter将代码文档分割为更小的节点,
    :param documents: LlamaIndex的Document对象列表
    :return: 分割后的节点列表
    """
    splitter = CodeSplitter(
        language="python",  # 指定代码语言
        chunk_size=1000,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(documents)  # 输入需是代码文件
    return nodes

from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import Document, Node
class MyCustomSplitter(NodeParser):
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def get_nodes_from_documents(self, documents: list[Document]) -> list[Node]:
        nodes = []
        for doc in documents:
            text = doc.text

            # 自定义分割逻辑：按固定字符数分割
            chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            # 以上是简单示例，可根据需要实现更复杂的分割逻辑
            
            for chunk in chunks:
                nodes.append(Node(text=chunk, metadata=doc.metadata))
        return nodes

# 语义分割
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.modelscope import ModelScopeEmbedding
def split_documents_semantic(documents, chunk_size:int=500, chunk_overlap:int=50):
    """
    使用语义分割器将文档分割为更小的节点
    :param documents: LlamaIndex的Document对象列表
    :return: 分割后的节点列表
    """
    # 使用ModelScope上成熟的中文嵌入模型
    embed_model = ModelScopeEmbedding(model_name="damo/nlp_corom_sentence-embedding_chinese-base")
    # 语义分割器
    semantic_parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        tokens_per_chunk=chunk_size,
        chunk_overlap=chunk_overlap,
        buffer_size=2,  # 计算相似度时的上下文窗口
        breakpoint_percentile_threshold=90  # 相似度低于95%分位则拆分
    )
    nodes = semantic_parser.get_nodes_from_documents(documents)
    return nodes

if __name__ == "__main__":
    # 读取文件
    documents = read_file_llama(document_dir="./file")
    nodes = split_documents_semantic(documents)

    print(nodes[:2])  # 打印前两个分割后的节点
    print(f"总节点数: {len(nodes)}")
    
    
    
    # 使用自定义分割器
    #splitter = MyCustomSplitter(chunk_size=500)
    #nodes = splitter.get_nodes_from_documents(documents)
    
