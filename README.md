# BGE-FAISS RAG — 检索增强问答系统

一个小而完整的 RAG（Retrieval-Augmented Generation）实现：把 8 篇技术文档切块、用 BGE 编码成向量存入 FAISS，
提问时先检索证据、再让大模型**只依据证据**作答，并把来源和相似度一起返回。

> 这是硕士毕业设计项目的可复现工程版本。原始实验见论文
> *Design, Implementation and Evaluation of a BGE-FAISS-Based RAG System for Domain-Specific Question Answering*（University of Portsmouth, 2026）。

## 它解决什么问题

大模型直接回答领域问题时，答案流畅但可能没有依据。RAG 的做法是：**先检索、后生成**，并且把检索到的证据一并返回，
让答案可以被逐条核对。本项目把这条链路做完整，并把"检索"单独暴露成 API，方便调试和评估。

## 架构

```
离线索引                          在线问答
────────────                      ─────────────────────────────────────
data/*.txt                        用户问题
   │                                 │
   ├─ 清洗 + 按 250 词切分            ├─ BGE 编码（同一模型）
   │   （重叠 50 词，最小 30 词）      │
   ├─ BGE 编码 → 归一化向量           ├─ FAISS 内积检索 Top-K
   │                                 │
   └─ FAISS IndexFlatIP ────────────→├─ 拼接带来源标注的上下文
                                     │
                                     ├─ DeepSeek 生成（只用上下文）
                                     │
                                     └─ 返回：答案 + 证据 + 相似度
```

选型说明：

- **IndexFlatIP + 归一化向量**：归一化后内积等价于余弦相似度，小语料下用精确检索，行为可解释、无需调参。
- **bge-small-en-v1.5**：384 维、检索导向、CPU 可跑，原型阶段性价比高。
- **切分 250 词 / 重叠 50 词**：兼顾单块语义完整与边界信息；重叠是为了避免答案正好被切在切口上。

## 检索评测结果

`eval/questions.json` 里 8 道题覆盖全部 8 篇文档（事实、对比、解释、实现四类）。一道题算命中，
当且仅当 Top-K 中出现来自含答案文档的片段：

```
ID  type           rank  top1 source                 score
Q1  factual        1     rag_intro.txt               0.8353
Q2  factual        1     faiss_vector_search.txt     0.8051
Q3  factual        1     bge_embeddings.txt          0.8205
Q4  comparative    1     dense_retrieval.txt         0.8516
Q5  explanatory    1     llm_hallucination.txt       0.8761
Q6  implementation 1     prompt_construction.txt     0.8309
Q7  explanatory    1     rag_evaluation.txt          0.7516
Q8  implementation 1     system_workflow.txt         0.7814

Hit@1 = 1.000   Hit@3 = 1.000   Hit@5 = 1.000   MRR = 1.000
```

**怎么读这个结果**：Hit@1 = 1.0 不代表系统很强，而是这次语料只有 8 篇主题互不重叠的文档（共 17 个 chunk），
检索本身不构成难点。这个集合的真正用途是**回归测试**——改动切分大小、换嵌入模型、加重排之后，
跑一遍就能立刻看出检索有没有退化。

生成质量的评估用的是论文里的 5 项人工评分（正确性 / 证据支持 / 相关性 / 清晰度 / 检索质量，1–5 分），
8 道题平均 **4.6 / 5.0**，并对比了非 RAG 基线、Top-3 与 Top-5、不同切分大小。人工评分无法自动复现，
脚本里用 `--llm` 生成答案后保存，人工按同一套标准打分。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 建索引（会下载 BGE 模型，约 130 MB）
python scripts/build_index.py

# 2) 配置 API Key（只有生成答案需要；纯检索不需要）
cp .env.example .env      # 然后填入你自己的 DEEPSEEK_API_KEY

# 3a) 起后端
uvicorn backend.main:app --reload
#     http://127.0.0.1:8000/docs

# 3b) 或起界面
streamlit run frontend/app.py
```

### 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/retrieve` | 只返回检索证据（便于单独检查检索质量） |
| POST | `/ask` | 检索 + 生成，返回答案与证据 |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAISS and why is IndexFlatIP used?", "top_k": 5}'
```

### 跑评测

```bash
python eval/evaluate.py                 # 只测检索，秒级完成
python eval/evaluate.py --top-k 3       # 换 top-k 复测
python eval/evaluate.py --llm           # 额外生成答案，供人工打分
python scripts/build_index.py --chunk-size 150 --overlap 30   # 切分消融实验
```

## 目录结构

```
backend/     config / rag_engine（检索）/ llm_client（生成）/ main（API）
scripts/     build_index.py —— 切分、编码、建 FAISS 索引
frontend/    app.py —— Streamlit 界面
eval/        questions.json + evaluate.py —— 检索指标与答案导出
data/        8 篇语料
```

## 已知限制

- 语料只有 8 篇、17 个 chunk，规模不足以说明真实场景的检索表现。
- 只做了稠密检索，没有 BM25 混合检索、没有 rerank 重排。
- 生成质量依赖人工打分，自动化程度低。
- 依赖外部 LLM API，离线时只能使用 `/retrieve`。

## 许可

MIT
