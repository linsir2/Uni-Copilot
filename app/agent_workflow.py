import os
import httpx # [Fix 1] Use Async Client
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage

class AgentState(TypedDict):
    question: str 
    original_question: str 
    chat_history: List[ChatMessage]
    retrieved_nodes: List[NodeWithScore]
    grade_status: str 
    retry_count: int
    final_response: Any
    source: str # 'local' or 'web'

def create_graph_app(retriever, llm):
    """构建并编译 LangGraph 工作流"""

    # --- 辅助：构造评分 Prompt ---
    def get_grader_prompt(question, context):
        return (
            f"你是一名严格的评分员。请评估以下检索到的教材片段是否包含回答用户问题所需的信息。\n"
            f"问题: {question}\n\n"
            f"教材片段:\n{context}\n\n"
            f"评判标准：\n"
            f"1. 片段必须包含具体的定义、解释或数据。\n"
            f"2. 如果片段只是提到了关键词但没解释（如目录、索引），判为 no。\n"
            f"3. 即使只有部分相关，只要有用，判为 yes。\n\n"
            f"请只回复 'yes' 或 'no'。"
        )
    
    # --- 辅助：异步 Tavily 搜索 ---
    async def tavily_search(query: str):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("⚠️ [Tavily] 未配置 API Key，跳过。")
            return []
        
        print(f"🌍 [Tavily] 正在异步搜索: {query}")
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic", # 'advanced' is slower, 'basic' is faster
            "include_answer": True,
            "max_results": 3,
        }

        try:
            # [Fix 1] 使用 httpx 进行异步请求，防止阻塞 FastAPI
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url="https://api.tavily.com/search",
                    json=payload,
                    timeout=10.0
                )
                resp.raise_for_status()
                data = resp.json()

            nodes = []
            # 1. Tavily 直接生成的 AI 答案
            if data.get("answer"):
                nodes.append(
                    NodeWithScore(
                        node=TextNode(
                            text=f"【网络智能摘要】: {data['answer']}",
                            metadata={"file_name": "Web", "page": "AI Summary"}
                        ),
                        score=1.0,
                    )
                )
            
            # 2. 具体的搜索结果
            for result in data.get("results", []):
                content = f"{result['content']}\n(Source: {result['url']})"
                nodes.append(
                    NodeWithScore(
                        node=TextNode(
                            text=content,
                            metadata={"file_name": "Web", "page": "Link"}
                        ),
                        score=0.9,
                    )
                )
            
            return nodes
        
        except Exception as e:
            print(f"❌ Tavily 搜索异常: {e}")
            return []

    # --- Node 1: Retrieve ---
    async def retrieve_node(state: AgentState):
        print("🔍 [Agent] Retrieving...")
        question = state["question"]
        chat_history = state.get("chat_history", [])

        # Query Rewrite (Simplification)
        # Only rewrite if we have history and it's not a retry
        search_query = question
        if chat_history and state.get("retry_count", 0) == 0:
            history_txt = "\n".join([f"{m.role}: {m.content}" for m in chat_history[-2:]])
            prompt = (
                f"基于对话历史，将用户最新的问题改写为独立的搜索关键词。\n"
                f"历史: {history_txt}\n问题: {question}\n"
                f"输出(仅关键词):"
            )
            res = await llm.acomplete(prompt)
            search_query = res.text.strip()
            print(f"   -> 改写 Query: {search_query}")

        nodes = await retriever.aretrieve(search_query)
        print(f"   -> 检索到 {len(nodes)} 条")
        return {"retrieved_nodes": nodes, "source": "local", "question": search_query}
    
    # --- Node 2: Grade ---
    async def grade_node(state: AgentState):
        question = state["question"]
        nodes = state["retrieved_nodes"]

        if not nodes:
            return {"grade_status": "no"}
            
        # 预览前3条内容用于评分
        context_preview = "\n".join([n.node.get_content()[:200] for n in nodes[:3]])
        prompt = get_grader_prompt(question, context_preview)

        response = await llm.acomplete(prompt)
        score = response.text.strip().lower()
        status = "yes" if "yes" in score else "no"
        
        print(f"⚖️ [Agent] 评分: {status}")
        return {"grade_status": status}
    
    # --- Node 3: Rewrite (Loop) ---
    async def rewrite_node(state: AgentState):
        print("🔄 [Agent] 重写查询词...")
        question = state["question"]
        prompt = f"用户问题 '{question}' 在教材中未搜到。请尝试提取核心实体词，去除修饰词，重写查询。"
        res = await llm.acomplete(prompt)
        new_q = res.text.strip()
        print(f"   -> 新 Query: {new_q}")
        
        return {
            "question": new_q, 
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    # --- Node 4: Web Search ---
    async def web_search_node(state: AgentState):
        print("🌍 [Agent] 启动 Web Search...")
        nodes = await tavily_search(state["original_question"])
        return {"retrieved_nodes": nodes, "source": "web"}
    
    # --- Node 5: Generate (With Citations) ---
    async def generate_node(state: AgentState):
        print("✍️ [Agent] Generating...")
        nodes = state["retrieved_nodes"]
        source_type = state.get("source", "local")
        
        # [Fix 2 & 3] Context Injection Logic
        context_lines = []
        for i, n in enumerate(nodes):
            # Safe Metadata Access
            meta = n.node.metadata or {}
            file = meta.get("file_name", "教材")
            page = meta.get("page", "?")
            
            # Construct Citation Tag
            if source_type == "web":
                citation = "[Web]"
            else:
                citation = f"[{file} P{page}]"
            
            # Inject into text so LLM sees it
            text = n.node.get_content()
            context_lines.append(f"引用来源 {citation}:\n{text}\n")

        context_str = "\n".join(context_lines)
        
        system_prompt = (
            f"你是一个专业的计算机助教。基于提供的{source_type}资料回答问题。\n"
            f"【引用规范】：\n"
            f"1. 凡是引用了资料里的观点或数据，必须在句尾加上来源标签，如 [计算机组成.pdf P12]。\n"
            f"2. 来源标签我已经都在资料里给你写好了，直接抄下来。\n"
            f"3. 如果资料里没有提及，不要编造。\n"
            f"4. 保持回答简洁、逻辑清晰。"
        )
        
        sys_msg = ChatMessage(role="system", content=system_prompt)
        user_msg = ChatMessage(role="user", content=f"资料：\n{context_str}\n\n问题：{state['original_question']}")
        
        # Return the stream iterator directly
        response_stream = await llm.astream_chat([sys_msg, user_msg])
        return {"final_response": response_stream}

    # --- Node 6: Fallback ---
    async def apologize_node(state: AgentState):
        return {"final_response": "抱歉，我在本地教材和网络上都未找到相关信息。"}

    # --- Build Graph ---
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("apologize", apologize_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("rewrite", "retrieve")

    # Conditional Logic
    def decide_local(state):
        if state["grade_status"] == "yes":
            return "generate"
        elif state["retry_count"] < 1: # Retry once
            return "rewrite"
        else:
            return "web_search"

    workflow.add_conditional_edges(
        "grade", 
        decide_local,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate") # Simplify: Trust web search for now
    workflow.add_edge("generate", END)
    workflow.add_edge("apologize", END)

    return workflow.compile()