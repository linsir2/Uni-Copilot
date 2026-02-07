from typing import Annotated, Dict, TypedDict, List, Any
from langgraph.graph import StateGraph, END
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import ChatMessage
from main import rag_engine

class AgentState(TypedDict):
    question: str # 当前的问题（可能是被重写过的版本）
    original_question: str # 最开始用户问的问题（没被llm修改过的版本）；用于最后llm生成回答
    chat_history: List[ChatMessage]
    retrieved_nodes: List[NodeWithScore]
    grade_status: str # "yes" or "no"
    retry_count: int
    final_response: Any

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
    
    async def retrieve_node(state: AgentState):
        print("🔍 [Node] Retrieving...")
        question = state["question"]
    
        nodes = await retriever.aretriever(question)

        print(f"   -> 检索到 {len(nodes)} 条相关片段")
        return {"retrieved_nodes": nodes}
    
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
    
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("rewrite", "retrieve")

    # --- 关键逻辑：条件边 ---
    def decide_next_step(state):
        status = state["grade_status"]
        retries = state.get("retry_count", 0)

        if status == "yes":
            return "generate" # 资料够了，去生成
        else:
            if retries < 1: # 🚨 最多重试 1 次，防止死循环
                return "rewrite"
            else:
                # 试过了还是不行，强行生成（或者这就该去 Tavily 了）
                print("🛑 [Agent] 重试次数耗尽，强行生成...")
                return "generate"

    workflow.add_conditional_edges(
        "grade",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    workflow.add_edge("generate", END)

    app = workflow.compile()
    return app