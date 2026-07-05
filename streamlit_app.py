"""
Streamlit + Groq API - 8種 RAG 策略 PDF 問答系統
安裝: pip install streamlit groq pypdf sentence-transformers numpy faiss-cpu scikit-learn
執行: streamlit run app.py
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np
import faiss
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ── 日誌設定 ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 常數 ─────────────────────────────────────────────────
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SMALL_CHUNK_SIZE = 300
SMALL_CHUNK_OVERLAP = 50
MAX_PDF_PAGES = 300

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

STRATEGY_DESC = {
    "1. 基礎語意搜尋": "快速、通用，直接用向量相似度找最相關片段。",
    "2. TF-IDF 關鍵詞": "精確匹配專有名詞、數字、術語，適合關鍵字型問題。",
    "3. 混合搜尋 (RRF)": "融合語意與關鍵詞排名，兼顧兩者優勢。",
    "4. 重新排序": "先粗篩再交給 LLM 精排，品質通常最高但較慢。",
    "5. 多查詢擴展": "自動改寫問題為多個角度，覆蓋面較廣。",
    "6. 上下文壓縮": "先檢索再用 LLM 濃縮成重點句，減少雜訊。",
    "7. 父子文檔": "用小片段精準定位，回傳完整大段落當上下文。",
    "8. 假設性答案 (HyDE)": "先生成假設答案再去比對，適合探索性問題。",
}


# ══════════════════════════════════════════════════════════
#  共用資源：Embedding 模型（跨 session 快取，只載入一次）
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 載入 Embedding 模型中（僅首次需要）…")
def load_embedding_model() -> SentenceTransformer:
    logger.info("載入 Embedding 模型中…")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ══════════════════════════════════════════════════════════
#  核心引擎：MultiStrategyRAG
# ══════════════════════════════════════════════════════════
class MultiStrategyRAG:
    """支援 8 種 RAG 策略的 PDF 問答引擎"""

    def __init__(self, embedding_model: SentenceTransformer):
        self.client: Optional[Groq] = None
        self.model_name: str = DEFAULT_MODEL
        self.embedding_model = embedding_model
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        # 父子文檔策略用的小片段索引（載入 PDF 時預先建好，避免每次查詢重算）
        self.small_chunks: list[str] = []
        self.small_index: Optional[faiss.IndexFlatIP] = None
        self.small_to_parent: dict[int, int] = {}
        self.pdf_loaded = False
        self.pdf_name: str = ""

    # ── 初始化 ────────────────────────────────────────────

    def set_api_key(self, api_key: str) -> str:
        api_key = api_key.strip()
        if not api_key:
            return "❌ 請輸入有效的 API Key"
        try:
            client = Groq(api_key=api_key)
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            self.client = client
            return "✅ API Key 驗證成功！"
        except Exception as e:
            self.client = None
            return f"❌ API Key 無效: {e}"

    # ── PDF 載入 ──────────────────────────────────────────

    def load_pdf(self, pdf_file, progress_cb=None) -> str:
        if pdf_file is None:
            return "⚠️ 請選擇 PDF 檔案"
        try:
            reader = PdfReader(pdf_file)
            if len(reader.pages) > MAX_PDF_PAGES:
                return f"❌ PDF 頁數超過上限（{MAX_PDF_PAGES} 頁），請拆分後再上傳"

            if progress_cb:
                progress_cb(0.15, "讀取 PDF 頁面…")

            pages_text: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            if not pages_text:
                return "❌ PDF 中未提取到任何文字（可能是掃描檔）"

            full_text = "\n".join(pages_text)

            if progress_cb:
                progress_cb(0.35, "分割文本片段…")
            self.chunks = self._split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not self.chunks:
                return "❌ 文本分割後無有效片段"

            if progress_cb:
                progress_cb(0.55, "生成向量嵌入…")
            raw_embeddings = self.embedding_model.encode(
                self.chunks, convert_to_numpy=True, show_progress_bar=False
            )
            self.embeddings = normalize(raw_embeddings, norm="l2").astype("float32")

            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

            if progress_cb:
                progress_cb(0.75, "建立 TF-IDF 索引…")
            self.tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

            if progress_cb:
                progress_cb(0.9, "建立父子文檔索引…")
            self._build_parent_child_index()

            self.pdf_loaded = True
            self.pdf_name = getattr(pdf_file, "name", "PDF 檔案")

            if progress_cb:
                progress_cb(1.0, "完成！")

            return (
                f"✅ 成功載入「{self.pdf_name}」！共 {len(reader.pages)} 頁，"
                f"提取 {len(pages_text)} 頁文字，分割為 {len(self.chunks)} 個片段"
            )

        except Exception as e:
            logger.exception("PDF 載入失敗")
            return f"❌ 載入失敗: {e}"

    def _build_parent_child_index(self):
        """預先建立小片段索引，並記錄「小片段 → 父片段」的映射（策略7用，避免每次查詢重算）"""
        full_text = " ".join(self.chunks)
        self.small_chunks = self._split_text(full_text, SMALL_CHUNK_SIZE, SMALL_CHUNK_OVERLAP)

        if not self.small_chunks:
            self.small_index = None
            return

        small_embeddings = self.embedding_model.encode(self.small_chunks, convert_to_numpy=True)
        small_embeddings = normalize(small_embeddings, norm="l2").astype("float32")

        self.small_index = faiss.IndexFlatIP(small_embeddings.shape[1])
        self.small_index.add(small_embeddings)

        # 建立映射：每個小片段對應到第一個包含它的大片段
        self.small_to_parent = {}
        for s_idx, small_chunk in enumerate(self.small_chunks):
            head = small_chunk[:60]
            for p_idx, parent in enumerate(self.chunks):
                if head in parent:
                    self.small_to_parent[s_idx] = p_idx
                    break

    # ── 文本分割 ──────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """依段落邊界分割文本，並在片段間保留重疊，避免截斷語意"""
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
                    # 保留前一段尾端文字作為重疊，維持語意連續性
                    tail = buffer[-overlap:] if overlap > 0 else ""
                    buffer = f"{tail} {para}".strip()
                else:
                    buffer = para

                while len(buffer) > chunk_size:
                    cut = buffer[:chunk_size].rfind("。")
                    if cut == -1:
                        cut = buffer[:chunk_size].rfind(". ")
                    if cut == -1:
                        cut = chunk_size
                    else:
                        cut += 1
                    merged.append(buffer[:cut].strip())
                    buffer = buffer[max(0, cut - overlap):].strip()

        if buffer:
            merged.append(buffer)

        return [c for c in merged if len(c) > 20]

    # ── LLM 呼叫 ──────────────────────────────────────────

    def _llm_call(self, prompt: str, system: str = "", max_tokens: int = 200,
                  temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _llm_call_stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
                         temperature: float = 0.3):
        """串流呼叫，逐段回傳文字（給 st.write_stream 用）"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── 向量搜尋 ──────────────────────────────────────────

    def _vector_search(self, query_text: str, top_k: int,
                       index: faiss.IndexFlatIP = None,
                       chunks: list[str] = None) -> list[tuple[int, float, str]]:
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
        results = self._vector_search(query, top_k)
        return [chunk for _, _, chunk in results]

    def strategy_2_tfidf(self, query: str, top_k: int = 3) -> list[str]:
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices if similarities[idx] > 0]

    def strategy_3_hybrid(self, query: str, top_k: int = 3) -> list[str]:
        k_rrf = 60
        sem_results = self._vector_search(query, top_k=min(top_k * 3, len(self.chunks)))
        sem_ranking = {idx: rank for rank, (idx, _, _) in enumerate(sem_results)}

        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        tfidf_ranking = {
            idx: rank for rank, idx in enumerate(tfidf_scores.argsort()[::-1][:top_k * 3])
        }

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
            reranked, seen = [], set()
            for n in numbers:
                if 0 <= n < len(candidates) and n not in seen:
                    reranked.append(candidates[n])
                    seen.add(n)
            for i, c in enumerate(candidates):
                if i not in seen:
                    reranked.append(c)
            return reranked[:top_k]
        except Exception:
            logger.warning("重新排序失敗，回退至基礎搜尋")
            return candidates[:top_k]

    def strategy_5_multi_query(self, query: str, top_k: int = 3) -> list[str]:
        expansion_prompt = f"請將以下問題改寫成 3 個不同角度的問題，每行一個，不要編號：\n{query}"
        try:
            result = self._llm_call(expansion_prompt, max_tokens=200, temperature=0.7)
            extra_queries = [
                q.strip().lstrip("0123456789.、-）) ")
                for q in result.split("\n") if q.strip()
            ][:3]
            queries = [query] + extra_queries
        except Exception:
            queries = [query]

        chunk_scores: dict[int, float] = {}
        k_rrf = 60
        for q in queries:
            results = self._vector_search(q, top_k=top_k)
            for rank, (idx, _, _) in enumerate(results):
                chunk_scores[idx] = chunk_scores.get(idx, 0) + 1.0 / (k_rrf + rank)

        sorted_indices = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]

    def strategy_6_contextual_compression(self, query: str, top_k: int = 3) -> list[str]:
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
        """父子文檔：小片段索引已在載入 PDF 時預先建好，這裡只需檢索 + 映射"""
        if self.small_index is None:
            return self.strategy_1_basic_similarity(query, top_k)

        results = self._vector_search(
            query, top_k=top_k * 2, index=self.small_index, chunks=self.small_chunks
        )

        parent_chunks, seen = [], set()
        for small_idx, _, _ in results:
            parent_idx = self.small_to_parent.get(small_idx)
            if parent_idx is not None and parent_idx not in seen:
                parent_chunks.append(self.chunks[parent_idx])
                seen.add(parent_idx)
            if len(parent_chunks) >= top_k:
                break

        return parent_chunks if parent_chunks else self.strategy_1_basic_similarity(query, top_k)

    def strategy_8_hypothetical_answer(self, query: str, top_k: int = 3) -> list[str]:
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

    # ── 檢索（不含生成，用於串流情境） ───────────────────

    def retrieve(self, query: str, strategy: str, top_k: int) -> list[str]:
        method_name = STRATEGY_MAP.get(strategy, "strategy_1_basic_similarity")
        retrieval_func = getattr(self, method_name)
        return retrieval_func(query, int(top_k))


# ══════════════════════════════════════════════════════════
#  Streamlit 介面
# ══════════════════════════════════════════════════════════

st.set_page_config(page_title="多策略 RAG PDF 問答系統", page_icon="🤖", layout="wide")


def get_rag() -> MultiStrategyRAG:
    """從 session_state 取出（或建立）當前使用者的 RAG 實例"""
    if "rag" not in st.session_state:
        embedding_model = load_embedding_model()
        st.session_state.rag = MultiStrategyRAG(embedding_model)
    return st.session_state.rag


def main():
    rag = get_rag()

    st.title("🤖 多策略 RAG PDF 問答系統")
    st.caption("採用 **8 種不同的 RAG 策略**，為您的 PDF 文件提供智能問答服務！")

    # ── 側邊欄：API Key、模型、PDF 上傳 ──
    with st.sidebar:
        st.header("🔑 設定")
        api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        rag.model_name = st.selectbox("使用模型", AVAILABLE_MODELS, index=0)

        if st.button("驗證 Key", use_container_width=True):
            with st.spinner("驗證中…"):
                status = rag.set_api_key(api_key_input)
            (st.success if status.startswith("✅") else st.error)(status)

        st.divider()

        st.header("📤 上傳 PDF")
        pdf_file = st.file_uploader("選擇 PDF 檔案", type=["pdf"])

        if st.button("🚀 載入文件", type="primary", use_container_width=True):
            if pdf_file is None:
                st.warning("⚠️ 請選擇 PDF 檔案")
            else:
                progress_bar = st.progress(0, text="準備中…")

                def progress_cb(pct: float, text: str):
                    progress_bar.progress(pct, text=text)

                status = rag.load_pdf(pdf_file, progress_cb=progress_cb)
                progress_bar.empty()
                (st.success if status.startswith("✅") else st.error)(status)

        if rag.pdf_loaded:
            st.info(f"📄 目前文件：{rag.pdf_name}（{len(rag.chunks)} 個片段）")

        st.divider()

        st.header("⚙️ RAG 策略")
        strategy = st.selectbox("選擇策略", list(STRATEGY_MAP.keys()))
        st.caption(STRATEGY_DESC.get(strategy, ""))
        top_k = st.slider("檢索片段數量 (Top-K)", min_value=1, max_value=10, value=3)

        with st.expander("📖 全部策略說明"):
            for name, desc in STRATEGY_DESC.items():
                st.markdown(f"**{name}**：{desc}")

    # ── 主畫面：提問區 ──
    st.subheader("💬 提問")

    example_questions = [
        "這份文件的主要內容是什麼？",
        "文件中提到哪些重要概念？",
        "有哪些關鍵數據或統計資料？",
        "文件的結論是什麼？",
    ]

    cols = st.columns(len(example_questions))
    for col, ex in zip(cols, example_questions):
        if col.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

    query = st.text_area(
        "輸入您的問題",
        key="query_input",
        placeholder="例如：這份文件的主要內容是什麼？",
        height=100,
    )

    ask_clicked = st.button("🔍 提問", type="primary", use_container_width=True)

    if ask_clicked:
        if not rag.client:
            st.error("❌ 請先在左側設定並驗證 API Key！")
        elif not rag.pdf_loaded:
            st.error("❌ 請先在左側上傳並載入 PDF 檔案！")
        elif not query.strip():
            st.warning("⚠️ 請輸入問題")
        else:
            with st.spinner(f"🔎 使用「{strategy}」檢索相關片段…"):
                try:
                    relevant_chunks = rag.retrieve(query, strategy, top_k)
                except Exception as e:
                    logger.exception("檢索失敗")
                    st.error(f"❌ 檢索失敗: {e}")
                    relevant_chunks = []

            if not relevant_chunks:
                st.warning("⚠️ 未檢索到相關片段，請嘗試其他策略或調整 Top-K")
            else:
                context = "\n\n---\n\n".join(relevant_chunks)

                st.subheader("💡 答案")
                try:
                    answer = st.write_stream(
                        rag._llm_call_stream(
                            prompt=(
                                f"請根據以下上下文回答問題。如果上下文中沒有相關資訊，請明確說明。\n\n"
                                f"上下文：\n{context}\n\n問題：{query}\n\n請用繁體中文詳細回答："
                            ),
                            system="你是專業的文件分析助手，回答時引用上下文中的具體內容。",
                            max_tokens=1024,
                            temperature=0.3,
                        )
                    )
                except Exception as e:
                    logger.exception("生成答案失敗")
                    st.error(f"❌ 生成答案失敗: {e}")

                with st.expander(f"📚 查看檢索到的文本片段（策略：{strategy}，共 {len(relevant_chunks)} 段）"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**片段 {i}**")
                        st.text(chunk)
                        st.divider()


if __name__ == "__main__":
    main()
