import httpx
import asyncio
import json
import logging
import re
import aiofiles
import os
import shutil
from datetime import datetime
from uuid import uuid4
from typing import List, Optional, Union, Dict, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage

# Try importing local reranker
try:
    from sentence_transformers import CrossEncoder
    HAS_LOCAL_RERANKER = True
except ImportError:
    HAS_LOCAL_RERANKER = False

logger = logging.getLogger("EduMatrixWorkflow")

# ================= Configuration =================
@dataclass
class WorkflowConfig:
    timeout_global: int = 120
    timeout_llm: int = 25
    timeout_search: int = 10
    max_retries: int = 2
    
    retrieve_top_k: int = 15
    rerank_top_n: int = 6
    local_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    web_search_depth: str = "basic"
    max_history_nodes: int = 8
    max_node_char_limit: int = 500
    log_file: str = "evolution_logs.jsonl"
    max_concurrent_llm_calls: int = 10
    
    all_strategies: List[str] = field(default_factory=lambda: ["narrow", "broaden", "keyword", "simplify"])
    threshold_map: Dict[str, int] = field(default_factory=lambda: {"FACT": 2, "COMPLEX": 2, "DEFAULT": 2})

    def get_threshold(self, query_type: str) -> int:
        return self.threshold_map.get(query_type, self.threshold_map["DEFAULT"])

# ================= Data Structures =================
@dataclass
class StrategyLog:
    trace_id: str
    timestamp: str
    original_query: str
    query_type: str
    strategy: str
    feedback: str
    score_before: int
    score_after: int
    delta: int

# ================= Events (Fixed Types) =================
class GradeEvent(Event):
    nodes: List[NodeWithScore] = field(default_factory=list)
    query: str = ""
    is_retry: bool = False
    # 修复: 允许 None
    strategy_used: Optional[str] = None

class RetryRequestEvent(Event):
    original_query: str = ""
    feedback: str = ""
    score: int = 0
    # 修复: 允许 None
    prev_strategy: Optional[str] = None
    best_node_idx: int = -1
    query_type: str = "DEFAULT"
    banned_strategies: List[str] = field(default_factory=list)

class RewriteEvent(Event):
    original_query: str = ""
    feedback: str = ""
    strategy: str = "default"
    system_hint: str = ""

class WebSearchEvent(Event):
    query: str = ""
    reason: str = ""

class GenerateEvent(Event):
    nodes: List[NodeWithScore] = field(default_factory=list)
    source: str = ""
    query: str = ""
    best_node_idx: int = -1

# ================= Helpers =================
class AtomicState:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    async def sadd(self, ctx: Context, key: str, value: str):
        async with self._locks[key]:
            current = await ctx.store.get(key, default=[])
            unique = set(current)
            if value not in unique:
                unique.add(value)
                await ctx.store.set(key, list(unique))

    async def rpush(self, ctx: Context, key: str, value: Any):
        async with self._locks[key]:
            current = await ctx.store.get(key, default=[])
            current.append(value)
            await ctx.store.set(key, current)
            
    async def get_list(self, ctx: Context, key: str) -> List[Any]:
         val = await ctx.store.get(key, default=[])
         return val if val is not None else []

class StrategyBrain:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.stats = defaultdict(lambda: defaultdict(lambda: {"runs": 0, "total_delta": 0, "wins": 0}))
        self._loaded = False
        self._lock = asyncio.Lock()

    async def ensure_loaded(self):
        if self._loaded or not os.path.exists(self.log_file): return
        async with self._lock:
            if self._loaded: return
            try:
                async with aiofiles.open(self.log_file, "r", encoding="utf-8") as f:
                    async for line in f:
                        try:
                            data = json.loads(line)
                            q_type = data.get("query_type", "DEFAULT")
                            strat = data.get("strategy", "default")
                            delta = data.get("delta", 0)
                            self.stats[q_type][strat]["runs"] += 1
                            self.stats[q_type][strat]["total_delta"] += delta
                            if delta > 0: self.stats[q_type][strat]["wins"] += 1
                        except: continue
                self._loaded = True
            except Exception as e: logger.warning(f"Brain load failed: {e}")

    async def get_advice(self, query_type: str) -> Dict[str, Any]:
        await self.ensure_loaded()
        if query_type not in self.stats: return {"recommend": [], "avoid": []}
        type_stats = self.stats[query_type]
        recommend, avoid = [], []
        for strat, s in type_stats.items():
            if s["runs"] < 3: continue
            avg_delta, win_rate = s["total_delta"] / s["runs"], s["wins"] / s["runs"]
            if win_rate > 0.6 and avg_delta > 0.5: recommend.append(strat)
            if avg_delta < 0: avoid.append(strat)
        return {"recommend": recommend, "avoid": avoid}

class EvolutionLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file_lock = asyncio.Lock()

    async def log(self, entry: StrategyLog):
        today = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(self.filepath):
            try:
                file_date = datetime.fromtimestamp(os.path.getmtime(self.filepath)).strftime("%Y-%m-%d")
                if file_date != today:
                    shutil.move(self.filepath, f"{self.filepath}.{file_date}")
            except Exception: pass
            
        try:
            line = json.dumps(asdict(entry), ensure_ascii=False)
            async with self._file_lock:
                async with aiofiles.open(self.filepath, mode='a', encoding='utf-8') as f:
                    await f.write(line + "\n")
        except Exception as e: logger.warning(f"Log write failed: {e}")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    try: return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return {}

# ================= Workflow V8.1 =================
class EduMatrixWorkflow(Workflow):
    def __init__(self, retriever, llm, config: Optional[WorkflowConfig] = None, tavily_api_key: Optional[str] = None):
        self.config = config or WorkflowConfig()
        super().__init__(timeout=self.config.timeout_global)
        
        self.retriever = retriever
        self.llm = llm
        self.tavily_api_key = tavily_api_key
        self.enable_web_search = bool(tavily_api_key)
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        self.evo_logger = EvolutionLogger(self.config.log_file)
        self.brain = StrategyBrain(self.config.log_file)
        self.atom = AtomicState()
        
        self._trace_buffers: Dict[str, List[dict]] = defaultdict(list)
        self._llm_semaphore = asyncio.Semaphore(self.config.max_concurrent_llm_calls)
        
        self._reranker = None
        if HAS_LOCAL_RERANKER:
            try:
                self._reranker = CrossEncoder(self.config.local_rerank_model, max_length=512)
                logger.info(f"✅ Local Reranker Loaded: {self.config.local_rerank_model}")
            except Exception as e:
                logger.warning(f"❌ Local Reranker Load Failed: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is not None and not self._http_client.is_closed: return self._http_client
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=self.config.timeout_search)
            return self._http_client

    async def aclose(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # 修复: trace_id 允许 None, error_code 允许 None
    def _add_trace_local(self, trace_id: Optional[str], step_name: str, status: str, details: dict, error_code: Optional[str] = None):
        if not trace_id: return
        entry = {"step": step_name, "status": status, "timestamp": datetime.now().isoformat(), "details": details}
        if error_code: entry["error_code"] = error_code
        self._trace_buffers[trace_id].append(entry)

    async def _flush_trace(self, ctx: Context, trace_id: Optional[str]):
        if not trace_id or trace_id not in self._trace_buffers: return
        new_entries = self._trace_buffers.pop(trace_id)
        if not new_entries: return
        for entry in new_entries:
            await self.atom.rpush(ctx, "timeline", entry)

    def _safe_get_content(self, node_with_score: NodeWithScore) -> str:
        try:
            content = node_with_score.node.get_content()
            if not content: return "[Empty]"
            limit = self.config.max_node_char_limit
            if len(content) > limit:
                return f"{content[:limit//2]}...[TRUNCATED]...{content[-limit//2:]}"
            return content
        except Exception: return "[Error]"

    async def _llm_call(self, prompt: str, fallback: Optional[Dict[Any, Any]] = None, retries: int = 2) -> dict:
        async with self._llm_semaphore:
            for attempt in range(retries + 1):
                try:
                    # 修复: Base.py 报错的原因是这里传了 timeout，确保 LlamaIndex 版本兼容
                    # 如果 LlamaIndex 底层不支持 timeout，外层 asyncio.wait_for 是唯一解
                    res = await asyncio.wait_for(self.llm.acomplete(prompt), timeout=self.config.timeout_llm)
                    data = extract_json_from_text(res.text)
                    if data: return data
                except asyncio.TimeoutError:
                    if attempt == retries: logger.warning("LLM Timeout")
                except Exception as e:
                    logger.warning(f"LLM fail: {e}")
                if attempt < retries: await asyncio.sleep(0.5 * (attempt + 1))
            return fallback or {}

    async def _rerank_nodes(self, query: str, nodes: List[NodeWithScore], top_n: int) -> List[NodeWithScore]:
        if not nodes or len(nodes) <= top_n: return nodes

        # 1. Try Local Cross-Encoder
        if self._reranker:
            try:
                pairs = [(query, self._safe_get_content(n)) for n in nodes]
                scores = await asyncio.get_event_loop().run_in_executor(None, self._reranker.predict, pairs)
                
                for n, s in zip(nodes, scores):
                    n.score = float(s)
                
                # 修复: sorted 遇到 None score 会报错，加 default
                reranked = sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)
                return reranked[:top_n]
            except Exception as e:
                logger.error(f"Local Rerank Failed: {e}, fallback to LLM")

        # 2. Fallback to LLM
        candidates = []
        for i, n in enumerate(nodes):
            text = self._safe_get_content(n)[:150].replace("\n", " ")
            candidates.append(f"ID {i}: {text}")
        
        prompt = (
            f"Query: {query}\nCandidates:\n{chr(10).join(candidates)}\n\n"
            f"Task: Select top {top_n} relevant IDs. Rank high to low.\n"
            "Output JSON: {\"ranked_ids\": [id...]}"
        )
        data = await self._llm_call(prompt, fallback={"ranked_ids": []})
        ranked_ids = data.get("ranked_ids", [])
        
        reranked = []
        seen = set()
        
        # 修复: LLM 可能返回字符串列表，做类型转换
        valid_ids = []
        if isinstance(ranked_ids, list):
            for x in ranked_ids:
                try: valid_ids.append(int(x))
                except: pass

        for idx in valid_ids:
            if 0 <= idx < len(nodes):
                if idx not in seen:
                    reranked.append(nodes[idx])
                    seen.add(idx)
        
        for i, n in enumerate(nodes):
            if i not in seen:
                reranked.append(n)
                if len(reranked) >= top_n: break
        
        return reranked[:top_n]

    # --- Search ---
    async def _tavily_search(self, query: str) -> List[NodeWithScore]:
        if not self.enable_web_search or not self.tavily_api_key: return []
        for attempt in range(2):
            try:
                client = await self._get_client()
                resp = await client.post(
                    url="https://api.tavily.com/search",
                    json={"api_key": self.tavily_api_key, "query": query, "search_depth": self.config.web_search_depth, "include_answer": True, "max_results": 3},
                )
                resp.raise_for_status()
                data = resp.json()
                nodes = []
                if data.get("answer"): nodes.append(NodeWithScore(node=TextNode(text=f"[Web]: {data['answer']}", metadata={"source": "web"}), score=0.9))
                for res in data.get("results", []): nodes.append(NodeWithScore(node=TextNode(text=f"{res['content']}\nSrc: {res['url']}", metadata={"source": "web"}), score=0.8))
                return nodes
            except Exception: await asyncio.sleep(1)
        return []

    # --- Step 1: Retrieve ---
    @step
    async def retrieve(self, ctx: Context, ev: Union[StartEvent, RewriteEvent]) -> GradeEvent:
        trace_id = await ctx.store.get("trace_id", default=uuid4().hex[:8])

        if isinstance(ev, StartEvent):
            question = ev.get("question", "")
            if not question: return GradeEvent()
            
            await ctx.store.set("trace_id", trace_id)
            await ctx.store.set("original_question", question)
            await ctx.store.set("retry_count", 0)
            await ctx.store.set("timeline", []) 
            await ctx.store.set("last_score", None)
            await ctx.store.set("query_type", "DEFAULT")
            await ctx.store.set("banned_strategies", [])
            
            self._trace_buffers.pop(trace_id, None)
            search_query = question
            self._add_trace_local(trace_id, "Retrieve", "START", {"query": question})
            is_retry, strategy = False, None
        else:
            search_query = ev.original_query
            self._add_trace_local(trace_id, "Retrieve", "RETRY", {"new_query": search_query, "strategy": ev.strategy})
            is_retry, strategy = True, ev.strategy

        if not self.retriever: return GradeEvent(nodes=[], query=search_query)

        try:
            nodes = await asyncio.wait_for(self.retriever.aretrieve(search_query), timeout=self.config.timeout_search)
            
            if len(nodes) > 3:
                reranked = await self._rerank_nodes(search_query, nodes, self.config.rerank_top_n)
                self._add_trace_local(trace_id, "Rerank", "SUCCESS", {"original": len(nodes), "kept": len(reranked), "method": "Local" if self._reranker else "LLM"})
                nodes = reranked

            self._add_trace_local(trace_id, "Retrieve", "SUCCESS", {"count": len(nodes)})
            await self._flush_trace(ctx, trace_id)
            
            return GradeEvent(nodes=nodes, query=search_query, is_retry=is_retry, strategy_used=strategy)
        except asyncio.TimeoutError:
            self._add_trace_local(trace_id, "Retrieve", "FAIL", {"error": "Timeout"}, error_code="ERR_TIMEOUT")
            await self._flush_trace(ctx, trace_id)
            return GradeEvent(nodes=[], query=search_query)

    # --- Step 2: Grade ---
    @step
    async def grade(self, ctx: Context, ev: GradeEvent) -> Union[GenerateEvent, RetryRequestEvent, WebSearchEvent]:
        trace_id = await ctx.store.get("trace_id")
        nodes = ev.nodes or []

        if not nodes:
            self._add_trace_local(trace_id, "Grade", "SKIP", {"reason": "No nodes"}, error_code="NO_CONTENT")
            return await self._handle_retry(ctx, ev.query, "No content", score=0, trace_id=trace_id)

        preview = "\n".join([f"[{i}] {self._safe_get_content(n)}" for i, n in enumerate(nodes[:5])])

        prompt = (
            f"Query: {ev.query}\nContext Preview:\n{preview}\n\n"
            "1. Classify: 'FACT' or 'COMPLEX'.\n"
            "2. Score (Max-Pooling 0-3): 0=Noise, 1=Weak, 2=Useful, 3=Direct.\n"
            "3. Best Node Index (-1 if none).\n"
            "Return JSON: {\"query_type\": \"...\", \"score\": int, \"reason\": \"...\", \"best_node_idx\": int}\n"
            "'reason' should be concise (less than 30 words)"
        )
        
        data = await self._llm_call(prompt, fallback={"score": 0, "reason": "LLM Error", "best_node_idx": -1, "query_type": "DEFAULT"})
        score, reason, best_idx = data.get("score", 0), data.get("reason", "N/A"), data.get("best_node_idx", -1)
        query_type = data.get("query_type", "DEFAULT").upper()

        if not ev.is_retry: await ctx.store.set("query_type", query_type)
        else: query_type = await ctx.store.get("query_type", default=query_type)

        last_score = await ctx.store.get("last_score")
        score_delta = 0
        
        if last_score is not None and ev.is_retry:
            score_delta = score - last_score
            # 修复: ev.strategy_used 可能为 None，Log时给默认值
            strat_name = ev.strategy_used or "unknown"
            task = asyncio.create_task(self.evo_logger.log(StrategyLog(trace_id, datetime.now().isoformat(), await ctx.store.get("original_question"), query_type, strat_name, reason, last_score, score, score_delta)))
            task.add_done_callback(lambda t: logger.warning(f"Log failed") if t.exception() else None)

            if score_delta <= 0 and ev.strategy_used:
                await self.atom.sadd(ctx, "banned_strategies", ev.strategy_used)

        await ctx.store.set("last_score", score)
        await ctx.store.set("last_delta", score_delta)
        
        self._add_trace_local(trace_id, "Grade", "SUCCESS", {"score": score, "delta": score_delta}, error_code=None if score >= 2 else "LOW_SCORE")

        if score >= self.config.get_threshold(query_type):
            return GenerateEvent(nodes=nodes, source="local", query=ev.query, best_node_idx=best_idx)
        
        # 修复: Atomic Read
        banned = await self.atom.get_list(ctx, "banned_strategies")
        return await self._handle_retry(ctx, ev.query, reason, score, prev_strategy=ev.strategy_used, best_idx=best_idx, q_type=query_type, banned=banned, trace_id=trace_id)

    async def _handle_retry(
        self, 
        ctx: Context, 
        query: str, 
        reason: str, 
        score: int, 
        prev_strategy: Optional[str] = None, # 允许 None
        best_idx: int = -1, 
        q_type: str = "DEFAULT", 
        banned: Optional[List[str]] = None,  # 允许 None
        trace_id: Optional[str] = None       # 允许 None
    ):
        retry_count = await ctx.store.get("retry_count", default=0)
        last_delta = await ctx.store.get("last_delta", default=0)
        
        should_retry = True
        stop_reason = ""

        if retry_count >= self.config.max_retries:
            should_retry = False; stop_reason = "Max retries"
        elif retry_count > 0 and score == 0:
            should_retry = False; stop_reason = "Score stuck at 0"
        elif prev_strategy and last_delta <= 0 and retry_count > 1:
             should_retry = False; stop_reason = f"Strategy ineffective repeatedly"

        # 修复: trace_id 可能为 None 的安全处理 (虽不太可能)
        if should_retry:
            await ctx.store.set("retry_count", retry_count + 1)
            await self._flush_trace(ctx, trace_id)
            # 修复: banned 和 trace_id 传递给 RetryRequestEvent
            return RetryRequestEvent(original_query=query, feedback=reason, score=score, prev_strategy=prev_strategy, best_node_idx=best_idx, query_type=q_type, banned_strategies=banned or [])

        original_q = await ctx.store.get("original_question")
        self._add_trace_local(trace_id, "Decision", "FALLBACK", {"reason": stop_reason})
        await self._flush_trace(ctx, trace_id)

        if self.enable_web_search: return WebSearchEvent(query=original_q, reason=reason)
        else: return GenerateEvent(nodes=[], source="fallback", query=original_q)

    # --- Step 3: Rewrite ---
    @step
    async def rewrite(self, ctx: Context, ev: RetryRequestEvent) -> RewriteEvent:
        trace_id = await ctx.store.get("trace_id")
        advice = await self.brain.get_advice(ev.query_type)
        
        hist_avoid, curr_banned = set(advice["avoid"]), set(ev.banned_strategies)
        if ev.prev_strategy: curr_banned.add(ev.prev_strategy)
        
        valid = [s for s in self.config.all_strategies if s not in hist_avoid and s not in curr_banned]
        if not valid: valid = ["simplify", "broaden"] 

        sys_hint, rec = "", ""
        if ev.best_node_idx == -1: rec = "broaden"; sys_hint = "No nodes. Try 'broaden'."
        elif ev.score == 1: rec = "narrow"; sys_hint = "Weak. Try 'narrow'."
        
        if rec and rec in curr_banned: rec = ""; sys_hint = "Try new approach."
        if advice["recommend"]: sys_hint += f"\n(Winners: {', '.join(advice['recommend'])})"

        prompt = (
            f"Original: '{ev.original_query}'\nFeedback: {ev.feedback}\n{sys_hint}\n"
            f"Allowed: {', '.join(valid)}\n"
            "Output JSON: {\"strategy\": \"...\", \"new_query\": \"...\"}\n"
            "Historical data shows 'broaden' is 60% more effective than 'narrow' for descriptive queries."
        )
        
        data = await self._llm_call(prompt, fallback={"strategy": "default", "new_query": ev.original_query})
        new_q, strat = data.get("new_query", ev.original_query), data.get("strategy", "default")
        
        if strat not in valid and strat in curr_banned: strat = valid[0] if valid else "default"

        self._add_trace_local(trace_id, "Rewrite", "SUCCESS", {"strategy": strat, "valid": valid, "query": new_q})
        await self._flush_trace(ctx, trace_id)
        return RewriteEvent(original_query=new_q, feedback="refined", strategy=strat)

    # --- Step 4 & 5 ---
    @step
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> GenerateEvent:
        trace_id = await ctx.store.get("trace_id")
        self._add_trace_local(trace_id, "WebSearch", "START", {"query": ev.query})
        nodes = await self._tavily_search(ev.query)
        self._add_trace_local(trace_id, "WebSearch", "SUCCESS", {"count": len(nodes)})
        await self._flush_trace(ctx, trace_id)
        return GenerateEvent(nodes=nodes, source="web", query=ev.query)

    @step
    async def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:
        trace_id = await ctx.store.get("trace_id")
        self._add_trace_local(trace_id, "Generate", "START", {"source": ev.source})
        await self._flush_trace(ctx, trace_id)
        
        timeline = await ctx.store.get("timeline")
        nodes = ev.nodes or []
        original_q = await ctx.store.get("original_question")

        sorted_nodes = []
        if 0 <= ev.best_node_idx < len(nodes):
            sorted_nodes = [nodes[ev.best_node_idx]] + [n for i, n in enumerate(nodes) if i != ev.best_node_idx]
        else: sorted_nodes = nodes

        serialized_nodes, context_lines = [], []
        for i, n in enumerate(sorted_nodes[:self.config.max_history_nodes]):
            content = self._safe_get_content(n)
            meta = n.node.metadata or {}
            is_best = (i == 0 and ev.best_node_idx != -1 and ev.source != "web")
            marker = "[⭐ BEST MATCH] " if is_best else ""
            citation = "[Web]" if ev.source == "web" else f"[{meta.get('file_name', 'Doc')} P{meta.get('page_label', '?')}]"
            context_lines.append(f"{marker}Citation {citation}:\n{content}\n")
            serialized_nodes.append({"id": n.node.node_id, "text": content, "metadata": meta, "score": n.score, "source": ev.source, "is_best": is_best})

        if not serialized_nodes and ev.source != "fallback":
            return StopEvent(result={"final_response": "No relevant info found.", "retrieved_nodes": [], "debug_timeline": timeline})

        sys_msg = ChatMessage(role="system", content="Answer based on context. Cite sources. Focus on [⭐ BEST MATCH].")
        user_msg = ChatMessage(role="user", content=f"Context:\n{''.join(context_lines)}\n\nQuestion: {original_q}")

        try:
            stream = await self.llm.astream_chat([sys_msg, user_msg])
        except Exception as e:
            logger.exception("Generate Failed")
            return StopEvent(result={"final_response": f"Generation Error: {str(e)}", "retrieved_nodes": serialized_nodes, "debug_timeline": timeline})

        self._trace_buffers.pop(trace_id, None)
        return StopEvent(result={"final_response": stream, "retrieved_nodes": serialized_nodes, "debug_timeline": timeline})