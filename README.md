使用 Python 與 Streamlit 建立的多策略 RAG 文件問答系統，可將 PDF 文件轉換為可檢索的知識庫，並結合向量搜尋、關鍵字搜尋與大型語言模型產生回答。

## 專案功能

- 上傳與解析 PDF 文件
- 文件切割與文字區塊處理
- Sentence Transformers 文字向量化
- FAISS 向量相似度搜尋
- TF-IDF 關鍵字檢索
- 語意搜尋與混合搜尋
- 多策略 RAG 文件問答
- 保留對話紀錄與上下文
- 使用 Streamlit 建立操作介面
- 串接大型語言模型 API

## RAG 處理流程

```text
PDF 文件
   ↓
文字解析
   ↓
文件切割 Chunking
   ↓
Embedding 向量化
   ↓
FAISS／TF-IDF 建立索引
   ↓
檢索相關文件內容
   ↓
組合提示詞 Prompt
   ↓
大型語言模型產生回答
```

## 使用技術

- Python
- Streamlit
- FAISS
- Sentence Transformers
- TF-IDF
- scikit-learn
- PDF 文件解析
- Groq LLM API
- Retrieval-Augmented Generation

## 專案檔案

```text
multi-strategy-rag-system/
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## 執行方式

### 1. 下載專案

```bash
git clone https://github.com/st3325228228-cpu/multi-strategy-rag-system.git
cd multi-strategy-rag-system
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定 API Key

請將大型語言模型 API Key 設定為環境變數或 Streamlit Secrets。

```toml
GROQ_API_KEY = "your-api-key"
```

請勿將真實 API Key 上傳至 GitHub。

### 4. 啟動程式

```bash
streamlit run streamlit_app.py
```

## 線上展示

[開啟 Multi-Strategy RAG System](https://stsfew-rag.hf.space)

## 專案目的

本專案用於練習與展示以下能力：

- Python AI 應用開發
- PDF 文件處理
- Embedding 向量化
- 向量資料檢索
- 關鍵字與語意混合搜尋
- RAG 系統架構
- LLM API 串接
- Streamlit Web App 開發
- AI 應用部署

## 系統限制

- 回答品質會受到文件內容與檢索結果影響
- 掃描型 PDF 可能需要額外 OCR 處理
- 大型文件可能需要較長的索引建立時間
- 語言模型仍可能產生不正確內容
- 使用者應根據原始文件再次確認重要資訊

## 未來改進方向

- 加入 ChromaDB 或其他向量資料庫
- 增加 RAG 評估指標
- 加入來源段落與頁碼引用
- 支援更多文件格式
- 增加重新排序 Reranking
- 加入使用者登入與知識庫管理
- 使用 Docker 建立標準化部署環境
