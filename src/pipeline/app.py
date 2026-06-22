import streamlit as st
import re
import json
from pathlib import Path
from llama_index.core import Settings, PropertyGraphIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from kg_prompt import KG_QA_TEMPLATE

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).parent.parent.parent
DATA_DIR             = BASE_DIR / "data"
CONFIG_PATH          = BASE_DIR / "config.md"
KG_STORE_DIR         = DATA_DIR / "kg_store"
PARSED_DOCS_DIR      = DATA_DIR / "parsed_docs"
EVAL_RESULTS_DIR     = DATA_DIR / "eval_results"
KG_STORE_PATH          = KG_STORE_DIR / "kg_store_2docs.json"
KG_STORE_ONTOLOGY_PATH = KG_STORE_DIR / "kg_store_ontology_2docs.json"

st.set_page_config(page_title="주택금융공사 고객문의 응답 챗봇", page_icon=None, layout="wide")

@st.cache_resource
def load_api_key() -> str:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)
    if not match:
        st.error("config.md에서 API 키를 찾을 수 없습니다.")
        st.stop()
    return match.group()

@st.cache_resource
def init_models(api_key: str):
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

@st.cache_resource
def load_index(path: Path = KG_STORE_ONTOLOGY_PATH):
    if not path.exists():
        st.error(f"KG 파일이 존재하지 않습니다: {path}\n\n먼저 build_kg.py 로 KG를 빌드해주세요.")
        st.stop()

    data = json.loads(path.read_text(encoding="utf-8"))
    graph_store = SimplePropertyGraphStore.from_dict(data)
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
    )
    return index

@st.cache_resource
def load_vector_index():
    parsed_files = list(PARSED_DOCS_DIR.glob("*_parsed.txt"))
    if not parsed_files:
        st.error("파싱된 텍스트 문서가 없습니다.")
        st.stop()
    docs = SimpleDirectoryReader(input_files=parsed_files).load_data()
    return VectorStoreIndex.from_documents(docs)
OPT_NAIVE    = "① Naive RAG"
OPT_PLAIN_KG = "② 앙상블 (Naive + 일반 KG)"
OPT_ONT_KG   = "③ 앙상블 (Naive + 온톨로지 KG)"

def page_chatbot():
    st.title("주택금융공사 고객문의 응답 챗봇")

    retriever_option = st.radio(
        "검색 방식 선택",
        [OPT_NAIVE, OPT_PLAIN_KG, OPT_ONT_KG],
        horizontal=True
    )

    st.markdown(f"**(현재 방식: {retriever_option})**")

    with st.spinner("파이프라인 및 모델 로딩 중..."):
        api_key = load_api_key()
        init_models(api_key)

        if retriever_option == OPT_NAIVE:
            vector_index = load_vector_index()
            query_engine = vector_index.as_query_engine(similarity_top_k=3)

        elif retriever_option == OPT_PLAIN_KG:
            vector_index = load_vector_index()
            kg_index = load_index(KG_STORE_PATH)
            vector_retriever = vector_index.as_retriever(similarity_top_k=3)
            kg_retriever = kg_index.as_retriever(
                include_text=True,
                retriever_mode="hybrid",
                similarity_top_k=3
            )
            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, kg_retriever],
                similarity_top_k=3,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False
            )
            query_engine = RetrieverQueryEngine.from_args(
                fusion_retriever,
                text_qa_template=KG_QA_TEMPLATE,
            )

        else:  # OPT_ONT_KG
            vector_index = load_vector_index()
            kg_index = load_index(KG_STORE_ONTOLOGY_PATH)
            vector_retriever = vector_index.as_retriever(similarity_top_k=3)
            kg_retriever = kg_index.as_retriever(
                include_text=True,
                retriever_mode="hybrid",
                similarity_top_k=3
            )
            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, kg_retriever],
                similarity_top_k=3,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False
            )
            query_engine = RetrieverQueryEngine.from_args(
                fusion_retriever,
                text_qa_template=KG_QA_TEMPLATE,
            )

    if "messages" not in st.session_state:
        st.session_state.messages = {OPT_NAIVE: [], OPT_PLAIN_KG: [], OPT_ONT_KG: []}

    msgs = st.session_state.messages[retriever_option]

    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "source_nodes" in msg:
                with st.expander("추론 경로 및 참고 문서 보기"):
                    for idx, node_info in enumerate(msg["source_nodes"]):
                        st.markdown(f"**[Source {idx+1}] Score: {node_info['score']:.4f}**")
                        st.code(node_info['text'], language="text")

    user_query = st.chat_input("예) 신혼부부 다자녀가구일 때 우대금리는 얼마인가요?")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        msgs.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            search_desc = "문서를 탐색 중입니다..." if retriever_option == OPT_NAIVE else "지식 그래프와 문서를 탐색 중입니다..."
            with st.spinner(search_desc):
                response = query_engine.query(user_query)
                answer = response.response

                source_nodes_data = []
                for n in response.source_nodes:
                    node_text = n.node.get_content()
                    score = n.score if n.score else 0.0
                    source_nodes_data.append({"text": node_text, "score": score})

            st.markdown(answer)

            with st.expander("추론 경로 및 참고 문서 보기"):
                src_desc = "텍스트 청크" if retriever_option == OPT_NAIVE else "Knowledge Graph 트리플 + 텍스트 청크 (Hybrid)"
                st.info(f"이 답변은 아래의 {src_desc}를 기반으로 작성되었습니다.")
                for idx, node_info in enumerate(source_nodes_data):
                    st.markdown(f"**[Source {idx+1}] Score: {node_info['score']:.4f}**")
                    st.code(node_info['text'], language="text")

        msgs.append({
            "role": "assistant",
            "content": answer,
            "source_nodes": source_nodes_data
        })


def main():
    page_chatbot()


if __name__ == "__main__":
    main()
