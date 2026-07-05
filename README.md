# Multi-Strategy RAG System

使用 Python 與 Streamlit 建立的多策略 RAG PDF 文件問答系統。

使用者可以上傳 PDF 文件，系統會自動解析文字、切割文件、建立向量索引與關鍵字索引，並提供 8 種不同的 RAG 檢索策略，最後透過大型語言模型產生繁體中文回答。

## 線上展示

[開啟 Multi-Strategy RAG System](https://multi-strategy-rag-system-qr2ct9faarcup2shvqozjr.streamlit.app)

## GitHub Repository

[查看專案原始碼](https://github.com/st3325228228-cpu/multi-strategy-rag-system)

---

## 專案特色

- 支援上傳與解析 PDF 文件
- 自動進行文件切割 Chunking
- 使用 Sentence Transformers 產生文字向量
- 使用 FAISS 建立向量搜尋索引
- 使用 TF-IDF 建立關鍵字搜尋索引
- 提供 8 種不同的 RAG 檢索策略
- 可調整檢索片段數量 Top-K
- 使用 Groq API 串接大型語言模型
- 顯示回答所使用的文件片段
- 保留多輪對話紀錄
- 提供範例問題快速測試
- 使用 Streamlit 建立互動式操作介面

---

## 8 種 RAG 策略

| 編號 | 策略 | 說明 | 適用情境 |
|---:|---|---|---|
| 1 | 基礎語意搜尋 | 使用 Embedding 與 FAISS 搜尋語意相近的文件片段 | 一般文件問答 |
| 2 | TF-IDF 關鍵詞搜尋 | 根據詞頻與關鍵字相似度搜尋內容 | 專有名詞、精確詞彙 |
| 3 | 混合搜尋 RRF | 結合語意搜尋與關鍵字搜尋結果 | 兼顧語意與關鍵字 |
| 4 | 重新排序 Reranking | 先檢索候選內容，再使用 LLM 重新排序 | 對回答精確度要求較高 |
| 5 | 多查詢擴展 Multi-Query | 將原始問題改寫成多個不同查詢 | 問題較模糊或範圍較廣 |
| 6 | 上下文壓縮 | 從檢索片段中保留與問題最相關的內容 | 長文件或雜訊較多的文件 |
| 7 | 父子文檔 Parent-Child | 使用小片段精確搜尋，再回傳較完整的父文件 | 結構化或長篇文件 |
| 8 | HyDE 假設性答案 | 先產生假設答案，再使用答案進行向量搜尋 | 探索性或語意較抽象的問題 |

---

## 系統操作流程

```text
輸入 Groq API Key
        ↓
驗證 API Key
        ↓
上傳 PDF 文件
        ↓
使用 PyPDF 解析文字
        ↓
文件切割 Chunking
        ↓
Sentence Transformers 產生 Embedding
        ↓
建立 FAISS 向量索引
        ↓
建立 TF-IDF 關鍵字索引
        ↓
選擇 RAG 檢索策略
        ↓
輸入問題並設定 Top-K
        ↓
檢索相關文件片段
        ↓
組合 Context 與 Prompt
        ↓
Groq LLM 產生繁體中文回答
        ↓
顯示回答與檢索來源
```

---

## 系統架構

```text
┌─────────────────────┐
│     Streamlit UI    │
│ API Key／PDF／問題   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     PDF Processing  │
│  PyPDF 文字解析      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Chunking       │
│ 文件切割與重疊處理   │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────┐
│Embedding │ │  TF-IDF  │
│語意向量   │ │關鍵字索引 │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            │
┌──────────┐      │
│  FAISS   │      │
│向量索引   │      │
└────┬─────┘      │
     └──────┬─────┘
            ▼
┌─────────────────────┐
│  Multi-Strategy RAG │
│    8 種檢索策略      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│      Groq LLM       │
│  產生繁體中文回答    │
└─────────────────────┘
```

---

## 主要功能

### 1. Groq API Key 驗證

使用者可以在側邊欄輸入自己的 Groq API Key。

系統會先呼叫 Groq API 進行驗證，驗證成功後才能載入 PDF 與執行文件問答。

API Key 輸入框使用密碼模式顯示，不會直接顯示完整內容。

### 2. PDF 文件上傳

支援上傳 PDF 文件，系統會：

1. 讀取 PDF 頁面
2. 提取每一頁的文字
3. 合併文件內容
4. 切割成多個文字片段
5. 建立 Embedding
6. 建立 FAISS 與 TF-IDF 索引

### 3. 文件切割

目前程式使用以下設定：

```text
主要 Chunk Size：800
主要 Chunk Overlap：150

小型 Chunk Size：300
小型 Chunk Overlap：50
```

文件片段之間保留重疊內容，可降低重要語意剛好被切斷的情況。

### 4. 向量語意搜尋

系統使用 Sentence Transformers 將文件片段轉換為向量，再使用 FAISS 執行相似度搜尋。

目前使用的 Embedding 模型：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

此模型支援多語言文字，可處理中文與英文文件。

### 5. 關鍵字搜尋

系統同時建立 TF-IDF 索引，設定包括：

```text
最大特徵數：3000
N-gram Range：(1, 2)
```

適合搜尋：

- 公司名稱
- 專業術語
- 法規名稱
- 數字與代號
- 文件中的精確關鍵字

### 6. Top-K 設定

使用者可以調整要檢索的文件片段數量：

```text
最少：1
最多：10
預設：3
```

較小的 Top-K 可以減少雜訊，較大的 Top-K 可以提供更多上下文。

### 7. 對話紀錄

系統會在目前的 Streamlit Session 中保留：

- 使用者問題
- AI 回答
- 使用的 RAG 策略
- 檢索到的文件片段

也可以使用「清除對話歷史」按鈕重新開始。

### 8. 檢索來源顯示

每次回答後，使用者可以展開「檢索來源」，查看回答所使用的原始文件片段。

這有助於：

- 驗證回答依據
- 比較不同策略結果
- 觀察檢索品質
- 降低只看模型回答造成的誤判

---

## 使用技術

### 程式語言

- Python

### Web App

- Streamlit

### 大型語言模型

- Groq API
- Llama 3.1 8B Instant

### 文件處理

- PyPDF

### Embedding

- Sentence Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`

### 向量搜尋

- FAISS
- Cosine Similarity
- L2 Normalization

### 關鍵字搜尋

- TF-IDF
- scikit-learn

### 資料處理

- NumPy

### RAG 技術

- Semantic Search
- Keyword Search
- Reciprocal Rank Fusion
- Reranking
- Multi-Query Expansion
- Contextual Compression
- Parent-Child Retrieval
- HyDE

---

## 專案檔案

```text
multi-strategy-rag-system/
├── README.md
├── requirements.txt
└── streamlit_app.py
```

### 檔案說明

| 檔案 | 說明 |
|---|---|
| `streamlit_app.py` | Streamlit UI、PDF 處理、索引建立、8 種 RAG 策略與回答生成 |
| `requirements.txt` | Python 套件清單 |
| `README.md` | 專案說明文件 |

---

## Requirements

目前使用的主要套件：

```text
streamlit
groq
pypdf
faiss-cpu
sentence-transformers
numpy
scikit-learn
```

---

## 本機安裝

### 1. Clone Repository

```bash
git clone https://github.com/st3325228228-cpu/multi-strategy-rag-system.git
```

### 2. 進入專案資料夾

```bash
cd multi-strategy-rag-system
```

### 3. 建立虛擬環境

Windows：

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS／Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. 安裝套件

```bash
pip install -r requirements.txt
```

### 5. 啟動 Streamlit

```bash
streamlit run streamlit_app.py
```

啟動後，瀏覽器通常會自動開啟：

```text
http://localhost:8501
```

---

## 使用方式

### 步驟 1：取得 Groq API Key

先至 Groq 平台申請 API Key。

API Key 通常會以以下格式開頭：

```text
gsk_
```

請勿將真實 API Key：

- 上傳至 GitHub
- 寫入公開程式碼
- 放入 README
- 分享給其他人

### 步驟 2：輸入 API Key

在 Streamlit 側邊欄輸入 Groq API Key，然後按下：

```text
驗證 API Key
```

### 步驟 3：上傳 PDF

選擇 PDF 文件後，按下：

```text
載入文件
```

系統會執行：

- PDF 文字解析
- 文件切割
- Embedding 建立
- FAISS 索引建立
- TF-IDF 索引建立

### 步驟 4：選擇 RAG 策略

從下拉選單選擇其中一種策略：

```text
基礎語意搜尋
TF-IDF 關鍵詞
混合搜尋 RRF
重新排序
多查詢擴展
上下文壓縮
父子文檔
HyDE
```

### 步驟 5：設定 Top-K

選擇要提供給模型的文件片段數量。

建議初次使用：

```text
Top-K = 3
```

### 步驟 6：輸入問題

可以輸入例如：

```text
這份文件的主要內容是什麼？
```

```text
文件中提到哪些重要概念？
```

```text
有哪些關鍵數據或統計資料？
```

```text
文件的結論是什麼？
```

### 步驟 7：查看檢索來源

回答產生後，可以展開檢索來源區塊，查看系統實際檢索到的 PDF 文字片段。

---

## 不同策略的使用建議

### 基礎語意搜尋

適合：

- 一般自然語言問題
- 查詢概念相近內容
- 快速取得回答

### TF-IDF 關鍵詞搜尋

適合：

- 人名
- 公司名稱
- 法規名稱
- 專有名詞
- 精確數字或代號

### 混合搜尋 RRF

適合：

- 同時需要語意理解與關鍵字匹配
- 不確定該使用哪種檢索方式
- 大多數一般文件問答

### 重新排序

適合：

- 對回答品質要求較高
- 候選片段較多
- 需要進一步篩選相關內容

此策略可能需要額外 LLM 呼叫，因此速度可能較慢。

### 多查詢擴展

適合：

- 問題描述不明確
- 問題可能有多種說法
- 想提高檢索涵蓋範圍

### 上下文壓縮

適合：

- 文件內容較長
- 檢索片段包含大量不相關資訊
- 希望降低傳送給模型的雜訊

### 父子文檔

適合：

- 長篇文件
- 有明確段落結構的文件
- 需要精確搜尋及完整上下文

### HyDE

適合：

- 探索性問題
- 文件中的表達方式與使用者問題差異較大
- 直接用問題搜尋效果不佳

---

## 目前模型設定

```text
LLM Model：
llama-3.1-8b-instant

Embedding Model：
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

主要 Chunk Size：
800

主要 Chunk Overlap：
150

預設 Top-K：
3
```

---

## 系統限制

- 目前主要支援文字型 PDF
- 掃描型 PDF 無法直接提取文字
- 掃描文件需要額外加入 OCR
- 一次主要處理一份 PDF
- 文件索引目前儲存在記憶體中
- 重新啟動應用後需要重新上傳文件
- 大型文件可能需要較長處理時間
- Groq API 需要網路連線
- 部分策略會增加 API 呼叫次數
- 回答品質取決於文件內容與檢索結果
- 大型語言模型仍可能產生錯誤資訊
- 重要資訊應回到原始 PDF 確認

---

## 資安與隱私注意事項

- 不要上傳含有機密或個人敏感資料的文件
- 不要將 Groq API Key 寫入公開 Repository
- 不要在截圖中公開 API Key
- 公開部署時應考慮檔案保存與刪除機制
- 若處理企業文件，建議使用私有部署環境
- 正式環境應加入檔案大小與格式限制
- 正式環境應加入輸入驗證、例外處理及使用量限制

---

## 專案目的

本專案用於學習與展示以下能力：

- Python AI 應用開發
- Streamlit Web App 開發
- PDF 文件處理
- 文件切割 Chunking
- Embedding 向量化
- FAISS 向量搜尋
- TF-IDF 關鍵字搜尋
- 混合檢索
- 多策略 RAG 架構
- Groq API 串接
- LLM Prompt 設計
- 對話狀態管理
- AI 應用雲端部署

---

## 未來改進方向

- 支援同時上傳多份文件
- 支援 DOCX、TXT、CSV 與網頁內容
- 加入 OCR 處理掃描型 PDF
- 加入 ChromaDB、Qdrant 或其他向量資料庫
- 保存不同使用者的知識庫
- 加入文件刪除與重新建立索引功能
- 顯示 PDF 頁碼與來源引用
- 顯示每個片段的相似度分數
- 加入 Cross-Encoder Reranker
- 加入 RAGAS 或其他 RAG 評估工具
- 增加回答正確性與忠實度評估
- 加入使用者登入與權限管理
- 加入對話紀錄匯出
- 加入模型與 Embedding 模型切換
- 加入 API 使用量與錯誤監控
- 使用 SQLite 或 PostgreSQL 儲存資料
- 使用 FastAPI 拆分後端服務
- 使用 Docker 建立標準化部署環境
- 建立 CI/CD 自動測試與部署流程

---

## 注意事項

本專案為 RAG、LLM、文件檢索與 Streamlit 應用開發的學習作品。

系統產生的回答僅供參考，不應直接用於醫療、法律、金融或其他高風險決策。使用者應查閱原始文件並自行確認內容正確性。
