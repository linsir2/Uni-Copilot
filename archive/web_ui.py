# import gradio as gr
# import requests

# API_URL = "http://127.0.0.1:8000/api/chat"

# def chat_with_backend(message, history):
#     """
#     message: 用户当前输入的问题
#     history: 之前的对话历史 (Gradio 自动维护，但目前我们的 API 是单轮问答，暂不用它)
#     """
#     if not message:
#         return "请输入问题..."
    
#     try:
#         # 发送请求给 FastAPI
#         payload = {"query": message}
#         response = requests.post(API_URL, json=payload, timeout=300, stream=True,)

#         if response.status_code == 200:
#             partial_text = ""
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     # 把新收到的字拼接到已有文本上
#                     text_chunk = chunk.decode("utf-8")
#                     partial_text += text_chunk
#                 # yield 是 Gradio 实现打字机效果的关键
#                 yield partial_text
#         else:
#             return f"❌ 服务器报错: {response.text}"
            
#     except Exception as e:
#         return f"❌ 连接失败，请检查 uvicorn 是否启动。\n错误详情: {str(e)}"
    
# demo = gr.ChatInterface(
#     fn=chat_with_backend,
#     title="🎓 EduMatrix 智能助教",
#     description="基于 RAG + 知识图谱的计算机课程问答系统 (Powered by Qwen)",
#     examples=["自然语言处理包含什么？", "书中提到的NLP中较为前沿的技术有哪些？", "NLP的基本思想是什么"],
# )

# if __name__ == "__main__":
#     # server_name="0.0.0.0" 允许局域网访问
#     demo.launch(server_name="0.0.0.0", server_port=7860)

# import gradio as gr
# import requests

# # FastAPI 后端地址
# API_URL = "http://127.0.0.1:8000/api/chat"

# def chat_with_backend(message, history):
#     """
#     message: 用户当前输入的问题
#     history: 之前的对话历史 [[问, 答], [问, 答]]
#     """
#     if not message:
#         yield "请输入问题..."
#         return

#     # 1. 适配 ChatEngine：构造 messages 列表
#     # 后端现在需要完整的上下文，而不仅仅是 query
#     messages_payload = []

#     # 把 Gradio 的 history 转换成后端需要的 {"role": "...", "content": "..."}
#     # for human, ai in history:
#     #     messages_payload.append({"role": "user", "content": human})
#     #     messages_payload.append({"role": "assistant", "content": ai})
    
#     # 🔥 修复点：不要用 for human, ai in history
#     # 因为新版 Gradio 的 history 可能包含额外信息 (如 metadata)
#     # 我们改为 robust 的写法：只读前两个元素
#     # for item in history:
#     #     human = item[0] # 用户消息
#     #     ai = item[1]    # AI 消息
#     #     messages_payload.append({"role": "user", "content": human})
#     #     messages_payload.append({"role": "assistant", "content": ai})

#     # 遍历 history 中的每一项
#     for item in history:
#         # 情况 A: 如果 item 是字典 (Gradio 新版 / type="messages")
#         # 格式如: {'role': 'user', 'content': 'xxx', 'metadata': ...}
#         if isinstance(item, dict):
#             role = item.get("role")
#             content = item.get("content")
#             # 只有当 role 和 content 都存在时才添加，忽略 metadata 等杂项
#             if role and content:
#                 messages_payload.append({"role": role, "content": content})
        
#         # 情况 B: 如果 item 是列表或元组 (Gradio 旧版 / 默认格式)
#         # 格式如: ['用户的问题', 'AI的回答']
#         elif isinstance(item, (list, tuple)):
#             if len(item) >= 2:
#                 # item[0] 是用户, item[1] 是 AI
#                 user_msg = item[0]
#                 ai_msg = item[1]
#                 if user_msg:
#                     messages_payload.append({"role": "user", "content": user_msg})
#                 if ai_msg:
#                     messages_payload.append({"role": "assistant", "content": ai_msg})

#     # 加上当前用户这一句
#     messages_payload.append({"role": "user", "content": message})
    
#     # 构造请求体
#     payload = {"messages": messages_payload}

#     try:
#         # 2. 发送请求 (开启流式 stream=True)
#         # timeout 设置大一点，防止模型思考时间过长导致超时
#         response = requests.post(API_URL, json=payload, stream=True, timeout=60)
        
#         # 3. 处理响应
#         if response.status_code == 200:
#             partial_text = ""
#             # iter_content 是 requests 库提供的流式读取方法
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     # 解码并拼接
#                     text_chunk = chunk.decode("utf-8", errors="replace")
#                     partial_text += text_chunk
#                     # 🔥 关键点：用 yield 实时刷新前端
#                     yield partial_text
#         else:
#             # 🔥 关键点：这里必须用 yield，不能用 return！
#             error_msg = f"❌ 服务器报错 (状态码 {response.status_code}):\n{response.text}"
#             yield error_msg
            
#     except Exception as e:
#         # 🔥 关键点：这里也必须用 yield
#         yield f"❌ 连接失败，请检查 uvicorn 是否启动。\n错误详情: {str(e)}"

# # 创建聊天界面
# demo = gr.ChatInterface(
#     fn=chat_with_backend,
#     title="🎓 EduMatrix 智能助教 (ChatEngine版)",
#     description="基于 RAG + 知识图谱 + 对话记忆构建 (Powered by Qwen)",
#     examples=["自然语言处理包含什么？", "书中提到的NLP中较为前沿的技术有哪些？", "NLP的基本思想是什么"],
# )

# if __name__ == "__main__":
#     # 启动
#     print("🚀 前端已启动: http://localhost:7860")
#     demo.launch(server_name="0.0.0.0", server_port=7860)

import gradio as gr
import requests

# FastAPI 后端地址
API_URL = "http://127.0.0.1:8000/api/chat"

# 🧼 新增：清洗函数，专门处理 Gradio 的复杂 content 格式
def clean_content(content):
    """
    把 Gradio 返回的复杂结构 [{'text': 'abc', 'type': 'text'}]
    清洗成纯字符串 'abc'
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # 如果是列表，把里面所有 type='text' 的内容拼起来
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(text_parts)
        
    return str(content)

def chat_with_backend(message, history):
    """
    message: 用户当前输入的问题
    history: 之前的对话历史
    """
    if not message:
        yield "请输入问题..."
        return

    # ================= 🛡️ 核心修复：更强的数据清洗 =================
    messages_payload = []
    
    for item in history:
        # 1. 提取原始数据
        role = None
        raw_content = None
        
        if isinstance(item, dict): # Gradio 新格式
            role = item.get("role")
            raw_content = item.get("content")
        elif isinstance(item, (list, tuple)) and len(item) >= 2: # Gradio 旧格式
            # 这里需要注意：Gradio 旧格式 list[0] 是 user, list[1] 是 assistant
            # 但这里我们简化处理，假设 history 已经是标准化过的或者不兼容此逻辑
            # 为了保险，我们主要适配 dict 格式（因为你用的是 type="messages"）
            pass 

        # 2. 清洗 content (这是解决 422 报错的关键！)
        if role and raw_content:
            clean_text = clean_content(raw_content)
            if clean_text:
                messages_payload.append({"role": role, "content": clean_text})
    
    # 处理当前用户输入 (message 也可能是复杂的，清洗一下)
    current_msg_clean = clean_content(message)
    messages_payload.append({"role": "user", "content": current_msg_clean})
    # =================================================================

    # 构造请求体
    payload = {"messages": messages_payload}
    
    try:
        # 发送请求
        response = requests.post(API_URL, json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            partial_text = ""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text_chunk = chunk.decode("utf-8", errors="replace")
                    partial_text += text_chunk
                    yield partial_text
        else:
            yield f"❌ 服务器报错 (状态码 {response.status_code}):\n{response.text}"
            
    except Exception as e:
        yield f"❌ 连接失败，请检查 uvicorn 是否启动。\n错误详情: {str(e)}"

# 创建聊天界面
demo = gr.ChatInterface(
    fn=chat_with_backend,
    title="🎓 EduMatrix 智能助教 (最终修复版)",
    description="基于 RAG + 知识图谱 + 对话记忆构建 (Powered by Qwen)",
    examples=["神经网络包含什么？", "它有什么优缺点？", "死锁产生的必要条件是什么？"],
)

if __name__ == "__main__":
    print("🚀 前端已启动: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)