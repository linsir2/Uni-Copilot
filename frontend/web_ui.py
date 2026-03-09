import gradio as gr
import requests
import html
import json
import os
import tempfile
import traceback
from pyvis.network import Network

# ================= 配置 =================
# 确保这些地址与你的后端 main.py 启动地址一致
API_BASE = "http://127.0.0.1:8000"
API_CHAT_URL = f"{API_BASE}/api/chat"
API_GRAPH_URL = f"{API_BASE}/api/graph"
API_UPLOAD_URL = f"{API_BASE}/api/upload"

# ================= 样式 CSS =================
custom_css = """
.gradio-container { height: 100vh !important; }
#graph-container { height: 600px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }
#chat-window { height: 600px !important; overflow-y: auto; }
"""

# ================= 1. 上传功能 =================
def upload_file(files):
    """
    上传文件到后端进行 RAG 摄取
    """
    if not files:
        return "⚠️ 请先选择文件"
    
    # Gradio 的 file 可能是列表
    files = files if isinstance(files, list) else [files]
    
    results = []

    for file_obj in files:
        try:
            print(f"📤 [前端] 正在上传: {file_obj.name}")
        
            # 构造 multipart/form-data 请求
            with open(file_obj.name, "rb") as f:
                files_payload = {"file": (os.path.basename(file_obj.name), f, "application/pdf")}
                response = requests.post(API_UPLOAD_URL, files=files_payload, timeout=300)
        
            if response.status_code == 200:
                res_json = response.json()
                results.append(
                    f"✅ {os.path.basename(file_obj.name)} 上传成功\n"
                    f"   ↳ task_id: {res_json.get('task_id', 'N/A')}\n"
                    f"   ↳ 状态：后台处理中"
                )
            else:
                results.append(
                    f"❌ {os.path.basename(file_obj.name)} 上传失败 "
                    f"(Code {response.status_code})"
                )
            
        except Exception as e:
            results.append(
                f"❌ {os.path.basename(file_obj.name)} 上传异常: {str(e)}"
            )
    
    return "\n\n".join(results)
# ================= 2. 图谱生成功能 =================
def generate_graph_html(query):
    """
    调用 /api/graph 接口并渲染 Pyvis 图谱
    """
    if not query:
        return "<div style='padding:20px; text-align:center; color:#999'>等待查询...</div>"
    
    print(f"🎨 [前端] 请求图谱: {query}")
    try:
        payload = {"messages": [{"role": "user", "content": str(query)}]}
        response = requests.post(API_GRAPH_URL, json=payload, timeout=15)
        
        try:
            data = response.json()
        except:
            return f"<div>❌ 后端返回非 JSON 数据</div>"

        links = data.get("links", [])
        if not links:
            return f"<div style='padding:20px; text-align:center; color:#999'>📭 未找到与 '{query}' 相关的图谱实体。<br>尝试提问具体的概念，如'神经网络'、'Transformer'。</div>"

        # 使用 Pyvis 生成图谱
        net = Network(
            height="580px", 
            width="100%", 
            bgcolor="#ffffff", 
            notebook=False,
            cdn_resources="in_line"
        )
        
        # 优化物理引擎参数，防止节点乱跑
        net.force_atlas_2based(
            gravity=-50, 
            central_gravity=0.01, 
            spring_length=100, 
            spring_strength=0.08, 
            damping=0.4, 
            overlap=0
        )

        seen_nodes = set()
        for link in links:
            src = str(link.get("source", "未知"))
            tgt = str(link.get("target", "未知"))
            rel = str(link.get("label", "RELATED"))
            
            if src not in seen_nodes:
                net.add_node(src, label=src, title=src, color="#4ecdc4", size=20)
                seen_nodes.add(src)
            if tgt not in seen_nodes:
                net.add_node(tgt, label=tgt, title=tgt, color="#ff6b6b", size=20)
                seen_nodes.add(tgt)
            
            net.add_edge(src, tgt, title=rel, label=rel, color="#cccccc")

        # 保存为临时 HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w+", encoding="utf-8") as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name
        
        with open(tmp_path, "r", encoding="utf-8") as f:
            raw_html = f.read()
        
        os.unlink(tmp_path)
        
        # 转义 HTML 以便在 iframe 中显示
        iframe_html = html.escape(raw_html)
        return f'<iframe style="width: 100%; height: 600px; border: none;" srcdoc="{iframe_html}"></iframe>'

    except Exception as e:
        traceback.print_exc()
        return f"<div>❌ 图谱渲染出错: {str(e)}</div>"

# ================= 3. 聊天功能 =================
def chat_stream(message, history):
    """
    调用 /api/chat 接口 (流式响应)
    """
    if history is None:
        history = []

    if not message:
        yield history
        return

    # 1️⃣ 添加用户消息
    history.append({
        "role": "user",
        "content": message
    })

    # 2️⃣ 预先插入一个空 assistant 消息（用于流式更新）
    history.append({
        "role": "assistant",
        "content": ""
    })

    yield history

    try:
        # 3️⃣ 发送给后端（直接发送 messages 格式）
        response = requests.post(
            API_CHAT_URL,
            json={"messages": history[:-1]},  # 不包含空 assistant
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            history[-1]["content"] = f"❌ Error {response.status_code}: {response.text}"
            yield history
            return

        partial_text = ""

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                text_chunk = chunk.decode("utf-8", errors="ignore")
                partial_text += text_chunk
                history[-1]["content"] = partial_text
                yield history

    except Exception as e:
        history[-1]["content"] = f"❌ Connection Error: {str(e)}"
        yield history

# ================= UI 构建 =================
with gr.Blocks(title="EduMatrix Pro", css=custom_css) as demo:
    
    gr.Markdown(
        """
        # 🎓 EduMatrix Pro: 多模态智能助教
        <div style='color: gray; font-size: 0.9em'>
        基于 RAG (Graph + Vector) 的教科书问答系统。上传 PDF，构建图谱，获取精确答案与原始图片证据。
        </div>
        """
    )
    
    with gr.Row():
        # --- 左侧：设置与上传 ---
        with gr.Column(scale=2, min_width=250):
            gr.Markdown("### 📂 文档管理")
            file_input = gr.File(
                label="上传教科书 (PDF)", 
                file_types=[".pdf"],
                file_count="multiple"
            )
            upload_btn = gr.Button("🚀 开始上传与摄取", variant="primary")
            upload_status = gr.Textbox(label="系统状态", interactive=False, lines=4)

            gr.Markdown("---")
            gr.Markdown("### ℹ️ 使用说明")
            gr.Markdown(
                """
                1. 上传 PDF 文件。
                2. 等待后台摄取完成 (观察控制台日志)。
                3. 在右侧提问。
                4. 系统会自动展示相关知识图谱和**原始图片证据**。
                """
            )

        # --- 中间：聊天窗口 ---
        with gr.Column(scale=5):
            gr.Markdown("### 💬 智能问答 (含图片证据)")
            chatbot = gr.Chatbot(
                elem_id="chat-window",
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=EduMatrix"),
                height=650,
                render_markdown=True
            )
            
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False, 
                    placeholder="请输入关于书本的问题，例如：'什么是反向传播？'",
                    scale=8,
                    container=False
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
                
            clear_btn = gr.Button("🗑️ 清空对话", size="sm", variant="secondary")

        # --- 右侧：知识图谱 ---
        with gr.Column(scale=4):
            gr.Markdown("### 🕸️ 动态思维导图")
            graph_output = gr.HTML(
                elem_id="graph-container",
                value="<div style='height:100%; display:flex; align-items:center; justify-content:center; color:#ccc'>图谱将随问题自动生成</div>"
            )

    # ================= 事件绑定 =================
    
    # 1. 上传
    upload_btn.click(
        upload_file, 
        inputs=[file_input], 
        outputs=[upload_status]
    )

    # 2. 聊天 (提交 -> 聊天流 + 图谱生成 并行)
    def clear_textbox():
        return ""
    
    # 回车提交
    txt_input.submit(
        chat_stream, 
        inputs=[txt_input, chatbot], 
        outputs=[chatbot]
    ).then(
        clear_textbox, None, txt_input
    )
    txt_input.submit(
        generate_graph_html,
        inputs=[txt_input],
        outputs=[graph_output]
    )

    # 按钮提交
    submit_btn.click(
        chat_stream, 
        inputs=[txt_input, chatbot], 
        outputs=[chatbot]
    ).then(
        clear_textbox, None, txt_input
    )
    submit_btn.click(
        generate_graph_html,
        inputs=[txt_input],
        outputs=[graph_output]
    )

    # 清空
    clear_btn.click(lambda: [], None, chatbot)
    clear_btn.click(lambda: "", None, graph_output)

if __name__ == "__main__":
    print("🚀 EduMatrix 前端已启动: http://localhost:7860")
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)