from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage
import os
import requests


class AgentState(TypedDict):
    question: str # 当前的问题（可能是被重写过的版本）
    original_question: str # 最开始用户问的问题（没被llm修改过的版本）；用于最后llm生成回答
    chat_history: List[ChatMessage]
    retrieved_nodes: List[NodeWithScore]
    grade_status: str # "yes" or "no"
    retry_count: int
    final_response: Any
    source: str # local or web

def create_graph_app(retriever, llm):
    """构建并编译 LangGraph 工作流"""
    # 辅助工具：构造评分prompt
    def get_grader_prompt(question, context):
        return (
            f"你是一名评分员。请评估以下检索到的教材片段是否与用户的问题相关。\n"
            f"问题: {question}\n\n"
            f"教材片段:\n{context}\n\n"
            f"如果片段包含能回答问题的关键词或语义，请回复 'yes'，否则回复 'no'。\n"
            f"只回复 'yes' 或 'no'，不要废话。"
        )
    
    def tavily_search(query: str):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("⚠️ 未配置 TAVILY_API_KEY，跳过联网搜索。")
            return []
        
        print(f"🌍 [Tavily] 正在搜索互联网: {query}")
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 3,
        }

        try:
            response = requests.post(
                url=os.getenv("TAVILY_BASE_URL") or "",
                json=payload,
                timeout=20,
            )
            data = response.json()

            nodes = []

            if data.get("answer"):
                nodes.append(
                    NodeWithScore(
                        node=TextNode(text=f"【Tavily 智能摘要】: {data['answer']}"),
                        score=1.0,
                    )
                )
            
            for result in data.get("results", []):
                content = f"【来源: {result['title']}】\n{result['content']}\n(URL: {result['url']})"
                nodes.append(
                    NodeWithScore(
                        node=TextNode(text=content),
                        score=0.9,
                    )
                )
            
            return nodes
        
        except Exception as e:
            print(f"❌ Tavily 搜索失败: {e}")
            return []

    async def retrieve_node(state: AgentState):
        print("🔍 [Node] Retrieving...")
        question = state["question"]
    
        nodes = await retriever.aretrieve(question)

        print(f"   -> 检索到 {len(nodes)} 条相关片段")
        return {"retrieved_nodes": nodes, "source": "local"}
    
    async def grade_node(state: AgentState):
        print("⚖️ [Agent] 正在评估资料质量...")
        question = state["question"]
        nodes = state["retrieved_nodes"]

        # 1. 如果根本没搜到，直接判死刑
        if not nodes:
            return {"grade_status": "no"}
            
        # 2. 构造上下文供 LLM 判断
        # 取前 3 个片段的内容拼接，避免 token 爆炸
        context_preview = "\n".join([n.get_content()[:200] for n in nodes[:3]])

        prompt = get_grader_prompt(question, context_preview)

        # 3. 调用 LLM 进行二分类
        # 这里用 complete 而不是 stream，因为我们要拿结果做判断
        response = await llm.acomplete(prompt)
        score = response.text.strip().lower()

        if "yes" in score:
            status = "yes"
        else:
            status = "no"
        
        print(f"   -> 评分结果: {status}")
        return {"grade_status": status}
    
    async def rewrite_node(state: AgentState):
        print("🔄 [Agent] 资料不全，正在重写查询词...")
        question = state["question"]

        prompt = (
            f"用户的问题是：'{question}'。\n"
            f"目前的检索结果不佳。请根据语义把这个问题重写得更精准，或者是提取核心关键词。\n"
            f"只输出重写后的问题，不要解释。"
        )

        response = await llm.acomplete(prompt)
        new_question = response.text.strip()

        print(f"   -> 新问题: {new_question}")
        
        # 更新问题，并增加计数器
        return {
            "question": new_question, 
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    async def web_search_node(state: AgentState):
        print("🌍 [Agent] 本地彻底没戏了，启动 Deep Research (Tavily)...")
        query = state["original_question"]

        web_nodes = tavily_search(query)

        print(f"   -> 联网获取了 {len(web_nodes)} 条信息")
        # 覆盖掉之前的本地结果，因为本地的反正也没用
        return {"retrieved_nodes": web_nodes, "source": "web"}
    
    async def grade_web_node(state: AgentState):
        print("⚖️ [Agent] 正在审核网络搜索结果...")

        query = state["original_question"]
        nodes = state["retrieved_nodes"]

        if not nodes:
            print("   -> 网络搜索为空")
            return {"grade_status": "no"}
        
        context_preview = "\n".join([n.get_content()[:300] for n in nodes[:3]])
        prompt = get_grader_prompt(query, context_preview)

        response = await llm.acomplete(prompt)
        status = "yes" if "yes" in response.text.strip().lower() else "no"
        print(f"-> 网络评分结果: {status}")
        return {"grade_status": status}
    
    async def generate_node(state: AgentState):
        print("✍️ [Agent] 正在组织语言生成回答 (Async)...")
        final_question = state["original_question"]
        nodes = state["retrieved_nodes"]
        history = state.get("chat_history", [])

        # 1. 拼凑上下文
        context_str = "\n\n".join([f"---片段---\n{n.get_content()}" for n in nodes])
        
        # 2. 构造 Prompt
        system_msg = ChatMessage(role="system", content="你是一个专业的计算机课程助教。请根据提供的教材片段回答问题。如果片段中没有答案，请诚实告知。")
        user_msg = ChatMessage(role="user", content=f"参考资料：\n{context_str}\n\n用户问题：{final_question}")
        
        messages = [system_msg] + history + [user_msg]

        response_stream = await llm.astream_chat(messages)

        return {"final_response": response_stream}
    
    # 当全网都搜不到时，体面地结束
    async def apologize_node(state: AgentState):
        print("🛑 [Agent] 彻底放弃，执行 Fallback...")
        text = "非常抱歉，我在本地教材和互联网上都没有找到相关信息。这可能是一个非常生僻的知识点，建议您查阅更专业的学术文献。"
        # 直接返回字符串，main.py 也能处理
        return {"final_response": text}
    
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("grade_web", grade_web_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("apologize", apologize_node)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("web_search", "grade_web")

    # 条件边 1: 本地评分后
    def decide_local(state):
        if state["grade_status"] == "yes":
            return "generate"
        elif state.get("retry_count", 0) < 1:
            return "rewrite"
        else:
            return "web_search"
    
    workflow.add_conditional_edges(
        "grade", decide_local,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search",
        }
    )

    # 🆕 条件边 2: 网络评分后
    def decide_web(state):
        if state["grade_status"] == "yes":
            return "generate" # 网络结果靠谱，去生成
        else:
            return "apologize" # 网络结果也是垃圾，去道歉
    
    workflow.add_conditional_edges(
        "grade_web", decide_web,
        {
            "generate": "generate",
            "apologize": "apologize",
        }
    )

    workflow.add_edge("apologize", END)
    workflow.add_edge("generate", END)

    app = workflow.compile()
    return app