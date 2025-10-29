# main.py
# 使用智谱聊天模型进行RAG问答

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatHunyuan
import time
import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

def get_response_from_zhipu(messages,query:str) -> str:
    """
    使用智谱聊天模型生成响应
    :param messages: 消息列表，包含用户和系统消息
    :return: 模型生成的响应文本
    """
    #从环境变量加载智谱API密钥

    # 初始化智谱聊天模型
    chat_model = ChatZhipuAI(
        model="glm-4",  # 指定模型名称
        api_key=os.getenv('ZHIPUAI_API_KEY'),  # 使用
        temperature=0.7,         # 控制生成文本的随机性
    )
    import qdrant_emded_store
    similar_docs = qdrant_emded_store.search_similar(query, top_k=3)
    # 构建上下文信息
    context_texts = "\n\n".join([f"来源：{doc['source']}（第{doc['page']}页）\n内容：{doc['text']}" for doc in similar_docs])
    # 在系统消息中添加上下文
    system_context = f"你是一个知识丰富的助手。以下是与用户查询相关的上下文信息：\n\n{context_texts}\n\n请基于以上信息回答用户的问题。"
    messages.insert(0, SystemMessage(content=system_context))

    # 生成响应
    user_text = "请回答以下问题：" + query
    messages.append(HumanMessage(content=user_text))
    response = chat_model.invoke(messages)

    return response.content,messages



if __name__ == "__main__":

    # 测试示例
    user_query = "什么是操作票？"
    messages = []
    response_text,messages = get_response_from_zhipu(messages, user_query)
    print("智谱模型响应：")
    print(response_text)

    
    print("\n完整对话记录：")
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "系统"
        print(f"{role}：{msg.content}\n")
