import gradio as gr
import requests

# FastAPI 后端地址
API_URL = "http://127.0.0.1:8000/api/chat"

# 🧼 数据清洗工具：处理 Gradio 复杂格式
def clean_content(content):
    """
    把 Gradio 返回的复杂结构 [{'text': 'abc', 'type': 'text'}] 清洗成纯字符串
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(text_parts)
        
    return str(content)

def chat_with_backend(message, history):
    """
    与后端交互的主函数
    """
    if not message:
        yield "请输入问题..."
        return

    # 1. 构造 messages 列表 (带清洗逻辑)
    messages_payload = []
    
    for item in history:
        # 兼容 Gradio 新旧版本格式
        role = None
        raw_content = None
        
        if isinstance(item, dict): # 新版
            role = item.get("role")
            raw_content = item.get("content")
        elif isinstance(item, (list, tuple)) and len(item) >= 2: # 旧版
            # 简化处理，暂时跳过旧版解析，主要依赖新版 type="messages"
            pass 

        if role and raw_content:
            clean_text = clean_content(raw_content)
            if clean_text:
                messages_payload.append({"role": role, "content": clean_text})
    
    # 加入当前问题
    current_msg_clean = clean_content(message)
    messages_payload.append({"role": "user", "content": current_msg_clean})

    # 2. 发送请求
    payload = {"messages": messages_payload}
    
    try:
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
    title="🎓 EduMatrix 智能助教",
    description="基于 Hybrid RAG (Vector + Graph) + Memory 构建",
    examples=["神经网络包含什么？", "它有什么优缺点？", "死锁产生的必要条件是什么？"],
)

if __name__ == "__main__":
    print("🚀 前端已启动: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)