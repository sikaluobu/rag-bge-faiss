"""Streamlit demo: ask a question, see the answer and the evidence behind it."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.llm_client import generate_answer  # noqa: E402
from backend.rag_engine import RAGEngine  # noqa: E402

st.set_page_config(page_title="BGE-FAISS RAG", page_icon="🔎", layout="wide")


@st.cache_resource
def load_engine() -> RAGEngine:
    return RAGEngine()


st.title("BGE-FAISS RAG — 检索增强问答")
st.caption(
    "8 篇技术文档 · BGE 向量化 · FAISS `IndexFlatIP` 检索 · DeepSeek 生成。"
    "答案只用检索到的证据，并附来源与相似度。"
)

with st.sidebar:
    st.subheader("检索设置")
    top_k = st.slider("Top-K", 1, 10, config.TOP_K)
    show_evidence = st.checkbox("显示检索证据", value=True)
    st.divider()
    st.caption(f"Embedding：`{config.EMBEDDING_MODEL}`")
    st.caption(f"生成模型：`{config.LLM_MODEL}`")
    if not config.LLM_API_KEY:
        st.warning("未检测到 `DEEPSEEK_API_KEY`，只能查看检索结果，无法生成答案。")

try:
    engine = load_engine()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

query = st.text_input("问题", placeholder="例如：What is FAISS and why is IndexFlatIP used?")

if query:
    with st.spinner("检索中…"):
        retrieved = engine.retrieve(query, top_k=top_k)

    if show_evidence:
        st.subheader("检索证据")
        for i, chunk in enumerate(retrieved, start=1):
            source = chunk["meta"].get("source", "unknown")
            with st.expander(
                f"[{i}] {source} · chunk {chunk['meta'].get('chunk_id')} · "
                f"相似度 {chunk['score']:.4f}"
            ):
                st.write(chunk["text"])

    st.subheader("回答")
    if not config.LLM_API_KEY:
        st.info("设置 `DEEPSEEK_API_KEY` 后可生成答案。")
    else:
        with st.spinner("生成中…"):
            try:
                answer = generate_answer(engine.build_prompt(query, retrieved))
                st.markdown(answer)
            except RuntimeError as exc:
                st.error(str(exc))
