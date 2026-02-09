import os
import re
import fitz  # PyMuPDF
import pdfplumber
import dashscope
from pathlib import Path
from http import HTTPStatus
from llama_index.core import Document
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import base64
import hashlib
import math
import tempfile
import time
import io
from PIL import Image
from collections import Counter

class LocalPDFParser:
    def __init__(self, 
                 pdf_path, 
                 image_output_dir="../data/parsed_images", 
                 cache_file="../data/vlm_cache.json", 
                 hash_record_file="../data/processed_hashes.json",
                 # [新增] Sidecar 文件路径，用于存放原本要把 metadata 撑爆的重型数据
                 sidecar_file="../data/page_heavy_data.json",
                 use_vlm=True, 
                 max_concurrency=5,
                 min_image_bytes=3072,
                 max_edge_size=1600,
                 jpeg_quality=85):
        
        self.pdf_path = Path(pdf_path)
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = Path(cache_file)
        self.hash_record_file = Path(hash_record_file)
        self.sidecar_file = Path(sidecar_file) # [新增]
        
        self.use_vlm = use_vlm
        self.model_name = "qwen-vl-plus" 
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.min_image_bytes = min_image_bytes
        self.max_edge_size = max_edge_size
        self.jpeg_quality = jpeg_quality

        self.cache_data = self._load_json(self.cache_file)
        self.global_processed_imgs = set(self._load_json(self.hash_record_file).get("hashes", []))
        
        # [新增] 加载或初始化 heavy data store
        self.page_heavy_data = self._load_json(self.sidecar_file)

    def _load_json(self, path: Path) -> Dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_data(self):
        """保存所有状态数据"""
        self._atomic_write(self.cache_file, self.cache_data)
        self._atomic_write(self.hash_record_file, {"hashes": list(self.global_processed_imgs)})
        self._atomic_write(self.sidecar_file, self.page_heavy_data) # [新增]

    def _atomic_write(self, path: Path, data: Any):
        try:
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, text=True)
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            print(f"⚠️ 数据保存失败 {path.name}: {e}")

    # ... (中间的 _compute_semantic_hash, _resize, _table_to_markdown, _extract_json 保持不变，省略以节省空间) ...
    def _compute_semantic_hash(self, img_bytes: bytes) -> str:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
                return hashlib.sha256(img.tobytes()).hexdigest()
        except Exception:
            return hashlib.md5(img_bytes).hexdigest()

    def _resize_and_compress_image(self, img_bytes: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                w, h = img.size
                if max(w, h) > self.max_edge_size:
                    scale = self.max_edge_size / max(w, h)
                    img = img.convert("RGB").resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=self.jpeg_quality)
                    return out.getvalue()
                return img_bytes
        except Exception:
            return img_bytes

    def _table_to_markdown(self, table):
        if not table or len(table) < 1: return ""
        cleaned = [[str(cell).replace('\n', ' ') if cell is not None else "" for cell in row] for row in table]
        header = "| " + " | ".join(cleaned[0]) + " |"
        sep = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
        rows = ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
        return f"\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n\n"

    def _extract_json(self, text: str) -> Dict:
        try: return json.loads(text)
        except: pass
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        return {"dense_caption": text, "entities": [], "relations": [], "keywords": [], "type": "UNKNOWN"}

    def _extract_text_keywords(self, text: str, top_k: int = 5) -> List[str]:
        keywords = []
        if not text: return []
        title = self._extract_section_title(text)
        if title:
            clean = re.sub(r"^[\d\.]+\s*|第[一二三四五六七八九十\d]+[章节]\s*", "", title).strip()
            if clean: keywords.append(clean)
        english_terms = re.findall(r'\b[A-Z][A-Za-z0-9_]{1,}\b', text)
        if english_terms:
            STOP_WORDS = {'THE', 'AND', 'FOR', 'FIG', 'TABLE', 'PAGE', 'SECTION'}
            filtered = [t for t in english_terms if t.upper() not in STOP_WORDS and len(t) > 1]
            keywords.extend(filtered)
        return keywords

    def _sanitize_canonical_name(self, name: str, canonical: str) -> str:
        if not canonical: return name
        clean_can = canonical.strip()
        clean_name = name.strip()
        if len(clean_can) > 40 or len(clean_can) > len(clean_name) * 4: return clean_name
        if " " in clean_can:
            words = clean_can.split()
            if len(set(words)) < len(words) * 0.5: return clean_name
        return clean_can

    def _dedup_graph_data(self, entities: List[Dict], relations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        unique_entities = []
        seen_entities = set()
        alias_map = {} 
        for e in entities:
            name = e.get("name", "").strip()
            raw_canonical = e.get("canonical_name", "").strip()
            category = e.get("category", "").strip()
            if not name: continue
            final_name = self._sanitize_canonical_name(name, raw_canonical)
            alias_map[name.upper()] = final_name
            if raw_canonical: alias_map[raw_canonical.upper()] = final_name
            key = (final_name.upper(), category.upper())
            if key not in seen_entities:
                seen_entities.add(key)
                e["name"] = final_name 
                unique_entities.append(e)
        unique_relations = []
        seen_relations = set()
        for r in relations:
            src = r.get("source", "").strip()
            tgt = r.get("target", "").strip()
            rel = r.get("relation", "").strip()
            if not src or not tgt: continue
            final_src = alias_map.get(src.upper(), src)
            final_tgt = alias_map.get(tgt.upper(), tgt)
            r["source"] = final_src
            r["target"] = final_tgt
            key = (final_src.upper(), final_tgt.upper(), rel.upper())
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)
        return unique_entities, unique_relations

    def _extract_section_title(self, text: str) -> Optional[str]:
        if not text: return None
        for line in text.split('\n')[:5]:
            line = line.strip()
            if re.match(r"^\d+(\.\d+)*\s+.+", line) or re.match(r"^第[一二三四五六七八九十\d]+[章节].+", line):
                return line
        return None

    async def _describe_image_with_retry(self, img_bytes: bytes, img_hash: str):
        if img_hash in self.cache_data: return self.cache_data[img_hash]
        if not self.use_vlm: return None
        processed_bytes = self._resize_and_compress_image(img_bytes)
        b64_data = base64.b64encode(processed_bytes).decode('utf-8')
        img_data_uri = f"data:image/jpeg;base64,{b64_data}"
        system_prompt = (
            "你是一个计算机组成原理课程的专家助教。请分析这张图片，为**向量检索**和**知识图谱构建**分别生成内容。\n"
            "请严格输出符合以下 Schema 的 JSON 格式：\n"
            "```json\n"
            "{\n"
            "  \"type\": \"(ARCH | FLOW | CODE | FORMULA | CHART | TABLE | OTHER)\",\n"
            "  \"dense_caption\": \"详细的自然语言描述，包含核心概念、数据流向和文字信息。\",\n"
            "  \"keywords\": [\"关键词1\", \"关键词2\"],\n"
            "  \"entities\": [{\"name\": \"如 ALU\", \"canonical_name\": \"如 Arithmetic Logic Unit\", \"category\": \"Component\"}],\n"
            "  \"relations\": [{\"source\": \"A\", \"target\": \"B\", \"relation\": \"FLOWS_TO\", \"description\": \"描述\"}],\n"
            "  \"ocr_text\": \"提取图中所有可见文字\"\n"
            "}\n"
            "```\n"
            "Canonical Name 用于消歧。无信息量返回: {\"type\": \"NONE\"}"
        )
        messages = [{"role": "user", "content": [{"image": img_data_uri}, {"text": system_prompt}]}]
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await asyncio.get_event_loop().run_in_executor(None, lambda: dashscope.MultiModalConversation.call(model=self.model_name, messages=messages, api_key=os.getenv("DASHSCOPE_API_KEY") or ""))
                    if response.status_code == HTTPStatus.OK: # type: ignore
                        content = response.output.choices[0].message.content # type: ignore
                        text = content[0]['text'] if isinstance(content, list) else str(content)
                        if text:
                            json_data = self._extract_json(text)
                            if json_data.get("type") != "NONE":
                                self.cache_data[img_hash] = json_data
                                return json_data
                        return None
                except Exception:
                    if attempt < 2: await asyncio.sleep(2 ** attempt)
            return None

    async def parse(self) -> List[Document]:
        print(f"🐢 [PageFirstParser v1.7 Sidecar] 开始解析: {self.pdf_path.name}")
        pages_content: Dict[int, Dict[str, Any]] = {}
        all_image_tasks = []
        doc_fitz = fitz.open(self.pdf_path)
        total_pages = len(doc_fitz)

        for i in range(total_pages):
            pages_content[i] = {'raw_text': "", 'text_parts': [], 'graph_entities': [], 'graph_relations': [], 'page_keywords': [], 'image_meta': [], 'evidence_images': []}

        print("    📖 正在提取文本与表格...")
        def extract_text():
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_content[i]['raw_text'] = text
                        pages_content[i]['text_parts'].append(text)
                    for idx, tbl in enumerate(page.extract_tables()):
                        md = self._table_to_markdown(tbl)
                        if md: pages_content[i]['text_parts'].append(f"\n=== 表格 {idx+1} ===\n{md}")
        await asyncio.get_event_loop().run_in_executor(None, extract_text)

        print("    🖼️ 正在提取图片...")
        for i in range(total_pages):
            page_fitz = doc_fitz[i]
            image_list = page_fitz.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                rects = page_fitz.get_image_rects(xref)
                bbox = [float(x) for x in rects[0]] if rects else []
                base_image = doc_fitz.extract_image(xref)
                img_bytes = base_image["image"]
                
                w, h = base_image.get("width", 0), base_image.get("height", 0)
                img_len = len(img_bytes)
                if len(img_bytes) < self.min_image_bytes or w < 200 or h < 200: continue

                aspect_ratio = w / h
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    continue

                byte_density = img_len / (w * h)
                if byte_density < 0.05:
                    continue

                if img_len < 10240:
                    continue
                
                img_hash = self._compute_semantic_hash(img_bytes)
                img_ext = base_image["ext"]
                img_path = self.image_output_dir / f"p{i+1}_{img_idx}_{img_hash[:8]}.{img_ext}"
                if not img_path.exists(): 
                    with open(img_path, "wb") as f: f.write(img_bytes)
                
                all_image_tasks.append(self._describe_image_with_retry(img_bytes, img_hash))
                pages_content[i]['image_meta'].append({"task_idx": len(all_image_tasks)-1, "path": str(img_path), "bbox": bbox, "hash": img_hash, "local_idx": img_idx})
        doc_fitz.close()

        image_results: List[Any] = [None] * len(all_image_tasks)
        if all_image_tasks:
            print(f"    🚀 正在处理 {len(all_image_tasks)} 张图片...")
            for i in range(0, len(all_image_tasks), 5):
                batch = all_image_tasks[i:i+5]
                results = await asyncio.gather(*batch, return_exceptions=True)
                for j, res in enumerate(results): image_results[i+j] = res
                if i+5 < len(all_image_tasks): await asyncio.sleep(1.0)

        final_documents = []
        current_section_title = "未知章节"
        for i in range(total_pages):
            page_data = pages_content[i]
            title = self._extract_section_title(page_data['raw_text'])
            if title: current_section_title = title
            
            text_kws = self._extract_text_keywords(page_data['raw_text'])
            page_data['page_keywords'].extend(text_kws)

            for meta in page_data['image_meta']:
                res = image_results[meta['task_idx']]
                if isinstance(res, Exception) or not res or res.get("type") == "NONE": continue
                
                page_data['text_parts'].append(f"\n=== 插图 {meta['local_idx']+1} ({res.get('type')}) ===\n描述: {res.get('dense_caption')}\nOCR: {res.get('ocr_text')}\n")
                if res.get("keywords"): page_data['page_keywords'].extend(res.get("keywords"))
                
                if meta['hash'] not in self.global_processed_imgs:
                    self.global_processed_imgs.add(meta['hash'])
                    if res.get("entities"):
                        for e in res.get("entities"): e.update({"source": "image", "page": i+1})
                        page_data['graph_entities'].extend(res.get("entities"))
                    if res.get("relations"):
                        for r in res.get("relations"): r.update({"provenance": "image", "page": i+1})
                        page_data['graph_relations'].extend(res.get("relations"))
                
                page_data['evidence_images'].append({"path": meta['path'], "bbox": meta['bbox'], "type": res.get("type")})

            kw_counter = Counter(page_data['page_keywords'])
            top_kws = [k for k, v in kw_counter.most_common(15)]
            keywords_str = ", ".join(top_kws)
            
            full_text = f"[SECTION] {current_section_title}\n[KEYWORDS] {keywords_str}\n=== 第 {i+1} 页 ===\n" + "\n".join(page_data['text_parts'])
            
            u_ents, u_rels = self._dedup_graph_data(page_data['graph_entities'], page_data['graph_relations'])
            
            # [关键修复] Sidecar Pattern: Metadata 瘦身
            # 生成一个唯一的 page_id
            page_id = f"{self.pdf_path.name}_p{i+1}"
            
            # 将“重数据”存入 Sidecar 字典，而不是 Document
            self.page_heavy_data[page_id] = {
                "graph_data": {"entities": u_ents, "relations": u_rels},
                "evidence_images": page_data['evidence_images']
            }

            final_documents.append(Document(
                text=full_text,
                metadata={
                    "file_name": self.pdf_path.name, 
                    "page_label": str(i+1), 
                    "section_title": current_section_title,
                    "content_type": "page_compound",
                    "page_id_key": page_id  # 只留一个索引 Key
                }
            ))

        self._save_data() # 保存 sidecar 文件
        print(f"✅ 解析完成！生成 {len(final_documents)} 个瘦身版文档片段。")
        return final_documents

async def load_pdf_locally(pdf_path):
    parser = LocalPDFParser(pdf_path)
    return await parser.parse()