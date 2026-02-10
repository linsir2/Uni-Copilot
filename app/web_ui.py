import gradio as gr
import requests
import html
from pyvis.network import Network
import tempfile
import os
import traceback

# ================= 配置 =================
API_CHAT_URL = "http://127.0.0.1:8000/api/chat"
API_GRAPH_URL = "http://127.0.0.1:8000/api/graph"

def clean_content(content):
    if content is None: return ""
    return str(content)

# 🔥 核心修复：解析 Gradio 的多模态列表
def parse_gradio_content(content):
    if content is None:
        return ""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)

# 🕸️ 1. 画图函数
def generate_graph_html(query):
    print(f"🎨 [前端] 准备请求图谱，Query: '{query}'")
    
    if not query: 
        print("⚠️ [前端] Query 为空，跳过画图")
        return "<div>请先提问...</div>"
    
    try:
        # 清洗 Query
        clean_query = parse_gradio_content(query)
        payload = {"messages": [{"role": "user", "content": clean_query}]}
        
        # 发送请求
        response = requests.post(API_GRAPH_URL, json=payload, timeout=10)
        
        try:
            data = response.json()
        except:
            return f"<div>❌ 后端返回异常: {response.text[:50]}...</div>"

        if data is None:
            return "<div>❌ 后端返回了空数据 (None)</div>"
            
        links = data.get("links", [])
        print(f"✅ [前端] 获取到 {len(links)} 条关系")
        
        if not links:
            return f"<div style='text-align:center; padding:20px; color: gray'>📭 关键词 '{clean_query}' 未找到相关图谱<br>(请尝试书中的核心概念，如：深度学习、神经网络)</div>"

        # 🔥🔥🔥 核心修复点 1：使用 cdn_resources='in_line' 🔥🔥🔥
        # 这会让 Pyvis 把所有 JS/CSS 直接写入 HTML，不依赖外部文件，iframe 才能显示！
        net = Network(
            height="500px", 
            width="100%", 
            bgcolor="#ffffff", 
            notebook=False,
            cdn_resources="in_line" 
        )
        
        # 构建图谱
        for link in links:
            # 确保转为字符串，防止 None 报错
            src = str(link.get("source", "未知"))
            tgt = str(link.get("target", "未知"))
            rel = str(link.get("label", "关联"))
            
            # 添加节点和边
            net.add_node(src, label=src, color="#4ecdc4", title=src)
            net.add_node(tgt, label=tgt, color="#ff6b6b", title=tgt)
            net.add_edge(src, tgt, title=rel, label=rel)

        # 布局算法
        net.force_atlas_2based()
        
        # 🔥🔥🔥 核心修复点 2：更稳健的文件写入 🔥🔥🔥
        try:
            # 使用临时文件保存 HTML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w+", encoding="utf-8") as tmp:
                # save_graph 在某些版本里可能不自动 flush，我们手动读
                net.save_graph(tmp.name)
                tmp_path = tmp.name
            
            # 重新读取内容
            with open(tmp_path, "r", encoding="utf-8") as f:
                raw_html = f.read()
            
            # 清理临时文件
            os.unlink(tmp_path)
            
        except Exception as e:
            print(f"❌ Pyvis 写文件失败: {traceback.format_exc()}")
            return f"<div>❌ Pyvis 渲染失败: {str(e)}</div>"
        
        # 转义 HTML 并返回 iframe
        escaped_html = html.escape(raw_html)
        return f'<iframe style="width: 100%; height: 500px; border: 1px solid #eee; border-radius: 8px;" srcdoc="{escaped_html}"></iframe>'

    except Exception as e:
        # 🔥 打印完整的错误堆栈，这样如果还有错，我们能看到具体是哪一行
        error_msg = traceback.format_exc()
        print(f"❌ [前端] 画图致命错误:\n{error_msg}")
        return f"<div>❌ 画图代码崩溃: {str(e)}</div>"

# 🗣️ 2. 聊天函数
def chat_with_backend(message, history):
    if history is None:
        history = []
        
    # 1. 构造发给后端的 API 格式
    messages_payload = []
    for msg in history:
        clean_text = parse_gradio_content(msg.get("content"))
        messages_payload.append({
            "role": msg.get("role"),
            "content": clean_text 
        })
    messages_payload.append({"role": "user", "content": message})

    # 2. UI 更新
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    
    # 🔥🔥🔥 核心修复在这里 🔥🔥🔥
    # 之前是 yield "", history (清空输入框)
    # 现在是 yield message, history (保留输入框内容)
    # 这样并行的 generate_graph_html 就能读到内容了！
    yield message, history

    # 3. 发送请求
    try:
        payload = {"messages": messages_payload}
        response = requests.post(API_CHAT_URL, json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            partial_text = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text_chunk = chunk.decode("utf-8", errors="replace")
                    partial_text += text_chunk
                    history[-1]['content'] = partial_text
                    
                    # 过程中保持输入框内容，防止误删
                    yield message, history 
        else:
            history[-1]['content'] = f"❌ Error {response.status_code}: {response.text}"
            yield message, history

    except Exception as e:
        history[-1]['content'] = f"❌ Connection Error: {str(e)}"
        yield message, history
    
    # 🔥🔥🔥 最后才清空输入框 🔥🔥🔥
    # 等所有事情都做完了，再把输入框变成空字符串
    yield "", history

# ================= UI 定义 =================
with gr.Blocks(title="EduMatrix Pro") as demo:
    gr.Markdown("# 🎓 EduMatrix: 多模态图谱智能助教")
    
    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=600, label="对话记录", value=[])
            
            with gr.Row():
                msg = gr.Textbox(show_label=False, placeholder="请输入问题...")
                submit_btn = gr.Button("发送", variant="primary")
            
            clear = gr.Button("清空对话")

        with gr.Column(scale=4):
            gr.Markdown("### 🕸️ 实时知识图谱")
            graph_view = gr.HTML(value="<div style='height:500px; border:1px dashed #ccc; padding:20px'>图谱将在这里显示...</div>")

    # --- 事件绑定 ---
    # 调整顺序：把画图放在前面触发（虽然并行，但好习惯）
    
    # 1. 触发画图
    msg.submit(generate_graph_html, inputs=[msg], outputs=[graph_view])
    submit_btn.click(generate_graph_html, inputs=[msg], outputs=[graph_view])

    # 2. 触发聊天
    msg.submit(chat_with_backend, inputs=[msg, chatbot], outputs=[msg, chatbot])
    submit_btn.click(chat_with_backend, inputs=[msg, chatbot], outputs=[msg, chatbot])
    
    # 清空
    def clear_history():
        return [], []
    clear.click(clear_history, outputs=[chatbot, graph_view])

if __name__ == "__main__":
    print("🚀 前端启动中 (Race Condition Fixed)...")
    demo.launch(server_name="0.0.0.0", server_port=7860)