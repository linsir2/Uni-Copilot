import os
import httpx
import asyncio
from uuid import uuid4
from typing import List, Any, Dict, Optional
from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, Workflow, step, Context
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage

# ================= 配置开关 =================
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
MAX_RETRIES = 1

# ================= 定义事件 (语义拆分) =================
class GradeEvent(Event):
    """检索完成，等待评分"""
    nodes: List[NodeWithScore]
    query: str

class RetryRequestEvent(Event):
    """评分不通过，请求重试（中间态）"""
    original_query: str
    feedback: str

class RewriteEvent(Event):
    """重写完成，携带新 Query（用于触发检索）"""
    original_query: str  # 这里的 semantic 是 "new query used for retrieval"
    feedback: str

class WebSearchEvent(Event):
    """本地重试耗尽，转网络"""
    query: str

class GenerateEvent(Event):
    """评分通过，准备生成"""
    nodes: List[NodeWithScore]
    source: str

# ================= 工作流定义 =================
class EduMatrixWorkflow(Workflow):
    def __init__(self, retriever, llm, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.retriever = retriever
        self.llm = llm
        
        # [并发安全] HTTP Client 懒加载
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is not None and not self._http_client.is_closed:
            return self._http_client 
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=10.0)
            return self._http_client

    async def aclose(self):
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _tavily_search(self, query: str) -> List[NodeWithScore]:
        if not ENABLE_WEB_SEARCH: return []
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key: return []
        
        try:
            client = await self._get_client()
            resp = await client.post(
                url="https://api.tavily.com/search",
                json={
                    "api_key": api_key, "query": query,
                    "search_depth": "basic", "include_answer": True, "max_results": 3
                }
            )
            resp.raise_for_status()
            data = resp.json()

            nodes = []
            if data.get("answer"):
                nodes.append(NodeWithScore(
                    node=TextNode(text=f"【网络摘要】: {data['answer']}", metadata={"file_name": "Web", "source": "web"}),
                    score=0.9 # 给高分，优先使用
                ))
            for res in data.get("results", []):
                nodes.append(NodeWithScore(
                    node=TextNode(text=f"{res['content']}\n(Source: {res['url']})", metadata={"file_name": "Web", "source": "web"}),
                    score=0.8
                ))
            return nodes
        except Exception as e:
            print(f"❌ Web Search Error: {e}")
            return []

    # --- Step 1: 检索 (监听 Start 或 Rewrite 完成事件) ---
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent | RewriteEvent) -> GradeEvent:
        trace_id = await ctx.get("trace_id", default=uuid4().hex[:8])
        
        if isinstance(ev, StartEvent):
            question = ev.get("question")
            await ctx.set("trace_id", trace_id) # type: ignore
            await ctx.set("original_question", question) # type: ignore
            await ctx.set("chat_history", ev.get("chat_history", [])) # type: ignore
            await ctx.set("retry_count", 0) # type: ignore
            search_query = question
            print(f"[{trace_id}] 🚀 Start: {search_query}")
        else:
            # 这里的 ev 是 RewriteEvent，携带的是已经改写好的新 query
            search_query = ev.original_query
            print(f"[{trace_id}] 🔄 Rewritten Retrieval: {search_query}")

        nodes = await self.retriever.aretrieve(search_query)
        # [优化] 限制上下文数量，防止 Token 爆炸
        return GradeEvent(nodes=nodes[:10], query=search_query)

    # --- Step 2: 评分 ---
    @step
    async def grade(self, ctx: Context, ev: GradeEvent) -> GenerateEvent | RetryRequestEvent | WebSearchEvent:
        trace_id = await ctx.get("trace_id")
        nodes = ev.nodes
        if not nodes:
            return await self._handle_retry(ctx, ev.query, "No content")

        preview = "\n".join([n.node.get_content()[:200] for n in nodes[:5]])
        # [优化] Prompt 约束，只输出 yes/no
        prompt = (
            f"问题: {ev.query}\n片段: {preview}\n"
            f"判断片段是否包含回答问题所需的信息。\n"
            f"规则：\n1. 包含定义、数据或解释 -> yes\n2. 仅提及关键词但无内容 -> no\n"
            f"请仅回答 'yes' 或 'no' (不要带标点)。"
        )
        res = await self.llm.acomplete(prompt)
        
        score_raw = res.text.strip().lower()
        # [优化] 更稳的判断
        is_relevant = score_raw == "yes" or score_raw.startswith("yes")
        
        if is_relevant:
            print(f"[{trace_id}] ✅ Grade Pass")
            return GenerateEvent(nodes=nodes, source="local")
        
        print(f"[{trace_id}] ❌ Grade Fail: {score_raw}")
        return await self._handle_retry(ctx, ev.query, "Irrelevant content")

    async def _handle_retry(self, ctx: Context, query: str, reason: str):
        retry_count = await ctx.get("retry_count")
        if retry_count < MAX_RETRIES:
            await ctx.set("retry_count", retry_count + 1) # type: ignore
            # [关键修复] 发出 RetryRequestEvent，而不是直接发 RewriteEvent，避免死循环
            return RetryRequestEvent(original_query=query, feedback=reason)
        return WebSearchEvent(query=await ctx.get("original_question"))

    # --- Step 3: 重写 (监听 RetryRequestEvent) ---
    @step
    async def rewrite(self, ctx: Context, ev: RetryRequestEvent) -> RewriteEvent:
        trace_id = await ctx.get("trace_id")
        print(f"[{trace_id}] 🧠 Rewriting query...")
        
        prompt = (
            f"原问题 '{ev.original_query}' 检索失败。\n"
            f"请提取核心实体，去除修饰词，生成一个新的搜索关键词。\n"
            f"仅输出关键词，不超过15字。"
        )
        res = await self.llm.acomplete(prompt)
        new_q = res.text.strip()
        
        # [关键] 返回 RewriteEvent，这个事件只被 Retrieve 监听
        return RewriteEvent(original_query=new_q, feedback="refined")

    # --- Step 4: 联网 ---
    @step
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> GenerateEvent:
        trace_id = await ctx.get("trace_id")
        print(f"[{trace_id}] 🌍 Web Fallback: {ev.query}")
        nodes = await self._tavily_search(ev.query)
        return GenerateEvent(nodes=nodes, source="web")

    # --- Step 5: 生成 ---
    @step
    async def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:
        nodes = ev.nodes
        original_q = await ctx.get("original_question")
        
        serialized_nodes = []
        context_lines = []
        
        # [优化] 再次限制进入 LLM 的片段数量，确保精简
        for n in nodes[:8]:
            meta = n.node.metadata or {}
            text = n.node.get_content()
            citation = "[Web]" if ev.source == "web" else f"[{meta.get('file_name','Doc')} P{meta.get('page','?')}]"
            context_lines.append(f"引用 {citation}:\n{text}\n")
            
            serialized_nodes.append({
                "text": text,
                "metadata": meta,
                "score": n.score
            })

        if not serialized_nodes:
            return StopEvent(result={"final_response": "未找到相关信息。", "retrieved_nodes": []})

        sys_msg = ChatMessage(role="system", content="基于资料回答。必须标注引用来源，如 [Doc P1]。")
        user_msg = ChatMessage(role="user", content=f"资料:\n{''.join(context_lines)}\n\n问题: {original_q}")
        
        stream = await self.llm.astream_chat([sys_msg, user_msg])
        
        return StopEvent(result={
            "final_response": stream, 
            "retrieved_nodes": serialized_nodes
        })

def create_graph_app(retriever, llm):
    return EduMatrixWorkflow(retriever=retriever, llm=llm, timeout=120, verbose=True)