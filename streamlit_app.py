"""
Gradio + Groq API - 8種 RAG 策略 PDF 問答系統 (優化版)
安裝: pip install gradio groq pypdf sentence-transformers numpy faiss-cpu scikit-learn
"""

import os
import logging
import re
from typing import Optional
from functools import lru_cache

import gradio as gr
import numpy as np
import faiss
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ── 日誌設定 ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 常數 ─────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SMALL_CHUNK_SIZE = 300
SMALL_CHUNK_OVERLAP = 50


class MultiStrategyRAG:
    """支援 8 種 RAG 策略的 PDF 問答引擎"""

    def __init__(self):
        self.client: Optional[Groq] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.pdf_loaded = False

    # ── 初始化 ────────────────────────────────────────────

    def set_api_key(self, api_key: str) -> str:
        """設定 Groq API Key"""
        api_key = api_key.strip()
        if not api_key:
            return "❌ 請輸入有效的 API Key"
        try:
            self.client = Groq(api_key=api_key)
            # 驗證 key 是否有效
            self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return "✅ API Key 驗證成功！"
        except Exception as e:
            self.client = None
            return f"❌ API Key 無效: {e}"

    def _ensure_embedding_model(self):
        """延遲載入 Embedding 模型（首次使用時才載入）"""
        if self.embedding_model is None:
            logger.info("載入 Embedding 模型中…")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding 模型載入完成")

    # ── PDF 載入 ──────────────────────────────────────────

    def load_pdf(self, pdf_file) -> str:
        if pdf_file is None:
            return "⚠️ 請選擇 PDF 檔案"
        try:
            self._ensure_embedding_model()
            reader = PdfReader(pdf_file)

            pages_text: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            if not pages_text:
                return "❌ PDF 中未提取到任何文字（可能是掃描檔）"

            full_text = "\n".join(pages_text)

            # 分割文本
            self.chunks = self._split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not self.chunks:
                return "❌ 文本分割後無有效片段"

            # 生成嵌入向量 & 建立 FAISS 索引（使用內積，需先正規化）
            raw_embeddings = self.embedding_model.encode(
                self.chunks, convert_to_numpy=True, show_progress_bar=True
            )
            self.embeddings = normalize(raw_embeddings, norm="l2").astype("float32")

            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # 內積 = cosine（已正規化）
            self.index.add(self.embeddings)

            # 建立 TF-IDF 索引
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=3000, ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

            self.pdf_loaded = True
            return (
                f"✅ 成功載入 PDF！共 {len(reader.pages)} 頁，"
                f"提取 {len(pages_text)} 頁文字，分割為 {len(self.chunks)} 個片段"
            )

        except Exception as e:
            logger.exception("PDF 載入失敗")
            return f"❌ 載入失敗: {e}"

    # ── 文本分割 ──────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """依句子邊界分割文本，避免截斷語意"""
        # 先按段落粗分
        paragraphs = re.split(r"\n{2,}", text)
        merged = []
        buffer = ""

        for para in paragraphs:
            para = re.sub(r"\s+", " ", para).strip()
            if not para:
                continue
            if len(buffer) + len(para) + 1 <= chunk_size:
                buffer = f"{buffer} {para}".strip()
            else:
                if buffer:
                    merged.append(buffer)
                # 若單一段落超過 chunk_size，則強制切割
                while len(para) > chunk_size:
                    # 嘗試在句號處切割
                    cut = para[:chunk_size].rfind("。")
                    if cut == -1:
                        cut = para[:chunk_size].rfind(". ")
                    if cut == -1:
                        cut = chunk_size
                    else:
                        cut += 1
                    merged.append(para[:cut].strip())
                    para = para[max(0, cut - overlap):].strip()
                buffer = para

        if buffer:
            merged.append(buffer)

        return [c for c in merged if len(c) > 20]  # 過濾過短片段

    # ── LLM 呼叫（統一封裝） ─────────────────────────────

    def _llm_call(self, prompt: str, system: str = "", max_tokens: int = 200,
                  temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    # ── 向量搜尋（共用） ─────────────────────────────────

    def _vector_search(self, query_text: str, top_k: int,
                       index: faiss.IndexFlatIP = None,
                       chunks: list[str] = None) -> list[tuple[int, float, str]]:
        """回傳 [(index, score, chunk), ...]"""
        index = index or self.index
        chunks = chunks or self.chunks

        query_vec = self.embedding_model.encode([query_text], convert_to_numpy=True)
        query_vec = normalize(query_vec, norm="l2").astype("float32")
        scores, indices = index.search(query_vec, min(top_k, len(chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks):
                results.append((int(idx), float(score), chunks[idx]))
        return results

    # ==================== 8 種 RAG 策略 ====================

    def strategy_1_basic_similarity(self, query: str, top_k: int = 3) -> list[str]:
        """策略1: 基礎語意相似度搜尋"""
        results = self._vector_search(query, top_k)
        return [chunk for _, _, chunk in results]

    def strategy_2_tfidf(self, query: str, top_k: int = 3) -> list[str]:
        """策略2: TF-IDF 關鍵詞搜尋"""
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices if similarities[idx] > 0]

    def strategy_3_hybrid(self, query: str, top_k: int = 3) -> list[str]:
        """策略3: 混合搜尋 — RRF (Reciprocal Rank Fusion)"""
        k_rrf = 60  # RRF 常數

        # 語意排名
        sem_results = self._vector_search(query, top_k=min(top_k * 3, len(self.chunks)))
        sem_ranking = {idx: rank for rank, (idx, _, _) in enumerate(sem_results)}

        # TF-IDF 排名
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        tfidf_ranking = {
            idx: rank
            for rank, idx in enumerate(tfidf_scores.argsort()[::-1][:top_k * 3])
        }

        # RRF 融合
        all_indices = set(sem_ranking.keys()) | set(tfidf_ranking.keys())
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in sem_ranking:
                score += 1.0 / (k_rrf + sem_ranking[idx])
            if idx in tfidf_ranking:
                score += 1.0 / (k_rrf + tfidf_ranking[idx])
            rrf_scores[idx] = score

        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]

    def strategy_4_reranking(self, query: str, top_k: int = 3) -> list[str]:
        """策略4: LLM 重新排序"""
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)

        prompt = (
            f"問題：{query}\n\n"
            "請對以下文本片段按照與問題的相關度排序（最相關的排前面），"
            "只回傳編號，用逗號分隔，例如：2,1,3\n\n"
        )
        for i, chunk in enumerate(candidates, 1):
            prompt += f"[{i}] {chunk[:200]}…\n\n"

        try:
            result = self._llm_call(prompt, max_tokens=50, temperature=0)
            numbers = [int(n) - 1 for n in re.findall(r"\d+", result)]
            reranked = []
            seen = set()
            for n in numbers:
                if 0 <= n < len(candidates) and n not in seen:
                    reranked.append(candidates[n])
                    seen.add(n)
            # 補齊未被排到的
            for i, c in enumerate(candidates):
                if i not in seen:
                    reranked.append(c)
            return reranked[:top_k]
        except Exception:
            logger.warning("重新排序失敗，回退至基礎搜尋")
            return candidates[:top_k]

    def strategy_5_multi_query(self, query: str, top_k: int = 3) -> list[str]:
        """策略5: 多查詢擴展"""
        expansion_prompt = (
            f"請將以下問題改寫成 3 個不同角度的問題，每行一個，不要編號：\n{query}"
        )
        try:
            result = self._llm_call(expansion_prompt, max_tokens=200, temperature=0.7)
            extra_queries = [
                q.strip().lstrip("0123456789.、-）) ")
                for q in result.split("\n")
                if q.strip()
            ][:3]
            queries = [query] + extra_queries
        except Exception:
            queries = [query]

        # 對每個查詢搜尋，用 RRF 融合
        chunk_scores: dict[int, float] = {}
        k_rrf = 60
        for q in queries:
            results = self._vector_search(q, top_k=top_k)
            for rank, (idx, _, _) in enumerate(results):
                chunk_scores[idx] = chunk_scores.get(idx, 0) + 1.0 / (k_rrf + rank)

        sorted_indices = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]

    def strategy_6_contextual_compression(self, query: str, top_k: int = 3) -> list[str]:
        """策略6: 上下文壓縮"""
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)

        compressed = []
        for chunk in chunks:
            try:
                result = self._llm_call(
                    f"從以下文本中，提取與問題「{query}」最直接相關的 1-3 句話。"
                    f"只輸出提取結果，不要加任何說明：\n\n{chunk}",
                    max_tokens=200,
                    temperature=0,
                )
                compressed.append(result if result else chunk[:300])
            except Exception:
                compressed.append(chunk[:300])

        return compressed

    def strategy_7_parent_child(self, query: str, top_k: int = 3) -> list[str]:
        """策略7: 父子文檔（小片段檢索 → 大上下文回傳）"""
        full_text = " ".join(self.chunks)
        small_chunks = self._split_text(full_text, SMALL_CHUNK_SIZE, SMALL_CHUNK_OVERLAP)

        if not small_chunks:
            return self.strategy_1_basic_similarity(query, top_k)

        small_embeddings = self.embedding_model.encode(small_chunks, convert_to_numpy=True)
        small_embeddings = normalize(small_embeddings, norm="l2").astype("float32")

        small_index = faiss.IndexFlatIP(small_embeddings.shape[1])
        small_index.add(small_embeddings)

        results = self._vector_search(query, top_k=top_k * 2, index=small_index, chunks=small_chunks)

        # 映射回原始大片段
        parent_chunks = []
        seen = set()
        for _, _, small_chunk in results:
            for i, big_chunk in enumerate(self.chunks):
                if i not in seen and small_chunk[:50] in big_chunk:
                    parent_chunks.append(big_chunk)
                    seen.add(i)
                    break
            if len(parent_chunks) >= top_k:
                break

        return parent_chunks if parent_chunks else self.strategy_1_basic_similarity(query, top_k)

    def strategy_8_hypothetical_answer(self, query: str, top_k: int = 3) -> list[str]:
        """策略8: HyDE — 假設性文檔嵌入"""
        try:
            hypothetical = self._llm_call(
                f"請針對以下問題，寫一段可能出現在文件中的回答段落（約 100 字）：\n{query}",
                max_tokens=200,
                temperature=0.7,
            )
        except Exception:
            hypothetical = query

        results = self._vector_search(hypothetical, top_k)
        return [chunk for _, _, chunk in results]

    # ── 主流程：生成答案 ─────────────────────────────────

    STRATEGY_MAP = {
        "1. 基礎語意搜尋": "strategy_1_basic_similarity",
        "2. TF-IDF 關鍵詞": "strategy_2_tfidf",
        "3. 混合搜尋 (RRF)": "strategy_3_hybrid",
        "4. 重新排序": "strategy_4_reranking",
        "5. 多查詢擴展": "strategy_5_multi_query",
        "6. 上下文壓縮": "strategy_6_contextual_compression",
        "7. 父子文檔": "strategy_7_parent_child",
        "8. 假設性答案 (HyDE)": "strategy_8_hypothetical_answer",
    }

    def generate_answer(self, query: str, strategy: str, top_k: int = 3):
        if not self.client:
            return "❌ 請先設定 API Key！", ""
        if not self.pdf_loaded:
            return "❌ 請先上傳 PDF 檔案！", ""
        if not query.strip():
            return "⚠️ 請輸入問題", ""

        try:
            method_name = self.STRATEGY_MAP.get(strategy, "strategy_1_basic_similarity")
            retrieval_func = getattr(self, method_name)
            relevant_chunks = retrieval_func(query, int(top_k))

            if not relevant_chunks:
                return "⚠️ 未檢索到相關片段，請嘗試其他策略或調整 Top-K", ""

            context = "\n\n---\n\n".join(relevant_chunks)

            answer = self._llm_call(
                prompt=(
                    f"請根據以下上下文回答問題。如果上下文中沒有相關資訊，請明確說明。\n\n"
                    f"上下文：\n{context}\n\n問題：{query}\n\n請用繁體中文詳細回答："
                ),
                system="你是專業的文件分析助手，回答時引用上下文中的具體內容。",
                max_tokens=1024,
                temperature=0.3,
            )

            source_info = (
                f"📚 使用策略：{strategy}\n"
                f"📄 檢索片段數：{len(relevant_chunks)}\n\n"
                f"{'=' * 50}\n相關文本片段：\n{'=' * 50}\n\n{context}"
            )

            return answer, source_info

        except Exception as e:
            logger.exception("生成答案失敗")
            return f"❌ 生成答案失敗: {e}", ""


# ══════════════════════════════════════════════════════════
#  Gradio 介面
# ══════════════════════════════════════════════════════════

def create_interface():
    rag = MultiStrategyRAG()

    def set_key(api_key: str) -> str:
        return rag.set_api_key(api_key)

    def upload_pdf(file) -> str:
        if file is None:
            return "⚠️ 請選擇 PDF 檔案"
        return rag.load_pdf(file.name)

    def ask_question(query, strategy, top_k):
        return rag.generate_answer(query, strategy, int(top_k))

    strategy_choices = list(MultiStrategyRAG.STRATEGY_MAP.keys())

    with gr.Blocks(title="🤖 多策略 RAG PDF 問答系統", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🤖 多策略 RAG PDF 問答系統\n"
            "採用 **8 種不同的 RAG 策略**，為您的 PDF 文件提供智能問答服務！"
        )

        # ── API Key ──
        with gr.Row():
            api_key_input = gr.Textbox(
                label="🔑 Groq API Key",
                placeholder="gsk_...",
                type="password",
                scale=3,
            )
            api_key_btn = gr.Button("驗證 Key", variant="secondary", scale=1)
            api_key_status = gr.Textbox(label="狀態", interactive=False, scale=2)

        api_key_btn.click(fn=set_key, inputs=[api_key_input], outputs=[api_key_status])

        gr.Markdown("---")

        with gr.Row():
            # ── 左側面板 ──
            with gr.Column(scale=1):
                gr.Markdown("### 📤 步驟 1: 上傳 PDF")
                pdf_input = gr.File(label="選擇 PDF 檔案", file_types=[".pdf"])
                upload_btn = gr.Button("🚀 載入文件", variant="primary")
                upload_status = gr.Textbox(label="載入狀態", interactive=False)

                gr.Markdown("### ⚙️ 步驟 2: 選擇 RAG 策略")
                strategy_dropdown = gr.Dropdown(
                    choices=strategy_choices,
                    value=strategy_choices[0],
                    label="RAG 策略",
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="檢索片段數量 (Top-K)",
                )

                gr.Markdown("""
### 📖 策略說明
| # | 策略 | 特點 |
|---|------|------|
| 1 | 基礎語意搜尋 | 快速、通用 |
| 2 | TF-IDF 關鍵詞 | 精確匹配專有名詞 |
| 3 | 混合搜尋 (RRF) | 兼顧語意與關鍵詞 |
| 4 | 重新排序 | LLM 精排，品質最高 |
| 5 | 多查詢擴展 | 覆蓋面廣 |
| 6 | 上下文壓縮 | 精簡雜訊 |
| 7 | 父子文檔 | 精準定位 + 完整上下文 |
| 8 | HyDE | 適合探索性問題 |
                """)

            # ── 右側面板 ──
            with gr.Column(scale=2):
                gr.Markdown("### 💬 步驟 3: 提問")
                question_input = gr.Textbox(
                    label="輸入您的問題",
                    placeholder="例如：這份文件的主要內容是什麼？",
                    lines=3,
                )
                ask_btn = gr.Button("🔍 提問", variant="primary", size="lg")

                gr.Markdown("### 💡 答案")
                answer_output = gr.Textbox(label="AI 回答", lines=10, interactive=False)

                with gr.Accordion("📚 查看檢索到的文本片段", open=False):
                    source_output = gr.Textbox(
                        label="相關來源", lines=15, interactive=False
                    )

        # ── 事件綁定 ──
        upload_btn.click(fn=upload_pdf, inputs=[pdf_input], outputs=[upload_status])
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, strategy_dropdown, top_k_slider],
            outputs=[answer_output, source_output],
        )
        question_input.submit(
            fn=ask_question,
            inputs=[question_input, strategy_dropdown, top_k_slider],
            outputs=[answer_output, source_output],
        )

        gr.Examples(
            examples=[
                ["這份文件的主要內容是什麼？"],
                ["文件中提到哪些重要概念？"],
                ["有哪些關鍵數據或統計資料？"],
                ["文件的結論是什麼？"],
            ],
            inputs=question_input,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0")
