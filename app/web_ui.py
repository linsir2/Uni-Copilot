import gradio as gr
import requests
import html
from pyvis.network import Network
import tempfile
import os

# ================= 配置 =================
API_CHAT_URL = "http://127.0.0.1:8000/api/chat"
API_GRAPH_URL = "http://127.0.0.1:8000/api/graph"

def clean_content(content):
    if content is None: return ""
    return str(content)

# 🕸️ 1. 画图函数
def generate_graph_html(query):
    if not query: return "<div>请先提问...</div>"
    
    try:
        # 1. 构造请求 (后端需要 list[dict])
        payload = {"messages": [{"role": "user", "content": query}]}
        response = requests.post(API_GRAPH_URL, json=payload, timeout=10)
        
        try:
            data = response.json()
        except:
            return f"<div>❌ 后端返回异常: {response.text[:50]}...</div>"

        if data is None:
            return "<div>❌ 后端返回了空数据 (None)</div>"
            
        links = data.get("links", [])
        
        if not links:
            return f"<div style='text-align:center; padding:20px; color: gray'>📭 关键词 '{query}' 未找到相关图谱<br>(请尝试书中的核心概念，如：深度学习、神经网络)</div>"

        # 2. 绘图逻辑 (Pyvis)
        # 注意：这里去掉了 font_color 参数，防止 Pylance 报错
        net = Network(height="500px", width="100%", bgcolor="#ffffff", notebook=False)
        
        for link in links:
            src = link.get("source", "未知")
            tgt = link.get("target", "未知")
            rel = link.get("label", "关联")
            
            net.add_node(src, label=src, color="#4ecdc4", title=src)
            net.add_node(tgt, label=tgt, color="#ff6b6b", title=tgt)
            net.add_edge(src, tgt, title=rel, label=rel)

        net.force_atlas_2based()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w+", encoding="utf-8") as tmp:
            net.save_graph(tmp.name)
            tmp.seek(0)
            raw_html = tmp.read()
        os.unlink(tmp.name)
        escaped_html = html.escape(raw_html)
        iframe_html = f'''
        <iframe 
            style="width: 100%; height: 500px; border: 1px solid #eee; border-radius: 8px;" 
            srcdoc="{escaped_html}">
        </iframe>
        '''
        return iframe_html

    except Exception as e:
        return f"<div>❌ 图谱生成代码出错: {str(e)}</div>"

# 🗣️ 2. 聊天函数 (🔥 关键：手动转换格式)
def chat_with_backend(message, history):
    # 【输入状态】
    # 因为没有 type="messages"，Gradio 传给我们的 history 绝对是 [[问, 答], [问, 答]]
    if history is None:
        history = []
        
    # 1. 格式转换：前端 List[List] -> 后端 List[Dict]
    messages_payload = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role and content:
            messages_payload.append({
                "role": role,
                "content": clean_content(content)
            })

    # 当前用户输入
    messages_payload.append({
        "role": "user",
        "content": clean_content(message)
    })

    # 2. 流式请求
    try:
        payload = {"messages": messages_payload}
        response = requests.post(API_CHAT_URL, json=payload, stream=True, timeout=60)
        
        partial_text = ""
        
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text_chunk = chunk.decode("utf-8", errors="replace")
                    partial_text += text_chunk
                    
                    # 🔥 【输出状态】
                    # 我们必须 yield List[Dict]，不然 Gradio 就会报 Data incompatible
                    # 这里的逻辑是：返回 旧历史 + [当前问, 当前生成的答]
                    yield history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": partial_text},
]
        else:
            yield history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": f"❌ Error {response.status_code}: {response.text}"}]

    except Exception as e:
        yield history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": f"❌ Connection Error: {str(e)}"}]

# ================= UI 定义 =================
with gr.Blocks(title="🎓 EduMatrix Pro") as demo:
    gr.Markdown("## 🎓 EduMatrix: 知识图谱智能助教")
    
    with gr.Row():
        with gr.Column(scale=6):
            # 🔥 绝对不加 type="messages"，这里是空的！
            # 这样它就会默认使用 List[List] 模式
            chatbot = gr.Chatbot(height=600)
            
            msg = gr.Textbox(label="你的问题", placeholder="试着问：什么是自然语言处理？")
            clear = gr.ClearButton([msg, chatbot])

        with gr.Column(scale=4):
            gr.Markdown("### 🕸️ 知识关联图谱")
            graph_view = gr.HTML(value="<div style='text-align:center; color:gray'>图谱将在这里显示...</div>")

    # 事件绑定
    msg.submit(generate_graph_html, inputs=[msg], outputs=[graph_view])
    
    # 聊天绑定
    msg.submit(
        chat_with_backend, 
        inputs=[msg, chatbot], # 传入旧历史 (List[List])
        outputs=[chatbot]      # 输出新历史 (List[List])
    ).then(
        lambda: "", outputs=[msg] # 清空输入框
    )

if __name__ == "__main__":
    print("🚀 前端启动中 (兼容模式)...")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme="soft")