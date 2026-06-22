"""
Phase 3 — LlamaIndex PropertyGraphIndex로 지식 그래프(KG) 자동 생성

[전체 흐름]
  1. config.md에서 OpenAI API 키를 읽어온다.
  2. LLM(GPT-4o-mini)과 임베딩 모델(text-embedding-3-small)을 초기화한다.
  3. data/ 폴더의 *_parsed.txt 파일들을 문서로 로드한다.
  4. LlamaIndex의 PropertyGraphIndex가 문서를 분석해 트리플(주어-관계-목적어) 형태의 KG를 생성한다. -> 이 클래스는 자동으로 LPG 방법으로 로드 됨.
  5. 생성된 KG를 data/kg_store.json에 UTF-8로 저장한다.

[사용 모델]
  - LLM      : gpt-4o-mini       — 문서에서 엔티티·관계 추출
  - Embedding: text-embedding-3-small — 노드·청크 벡터화 (유사도 검색용)

[입력]  data/*_parsed.txt   (Phase 2에서 PDF를 파싱한 텍스트)
[출력]  data/kg_store.json  (PropertyGraph 직렬화 JSON)
"""

import re
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,   # 로컬 파일을 Document 객체로 변환하는 유틸
    PropertyGraphIndex,      # 엔티티-관계 그래프를 자동 구축하는 인덱스
    Settings,                # LLM·임베딩 등 전역 설정 객체
)
from llama_index.core.graph_stores import SimplePropertyGraphStore  # 인메모리 그래프 저장소
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI                          # OpenAI LLM 래퍼
from llama_index.embeddings.openai import OpenAIEmbedding           # OpenAI 임베딩 래퍼

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
# __file__ 기준으로 프로젝트 루트를 계산해 경로를 하드코딩하지 않는다.
DATA_DIR        = Path(__file__).parent.parent.parent / "data"
CONFIG_PATH     = Path(__file__).parent.parent.parent / "config.md"
PARSED_DOCS_DIR = DATA_DIR / "parsed_docs"
KG_STORE_DIR    = DATA_DIR / "kg_store"
KG_STORE_PATH   = KG_STORE_DIR / "kg_store.json"


def load_api_key() -> str:
    """
    config.md에서 OpenAI API 키를 정규식으로 추출한다.

    config.md 형식 예시:
        GPT api key: sk-proj-xxxxxxxxxxxx...

    OpenAI 키는 'sk-' 로 시작하며 20자 이상의 영숫자/하이픈/언더스코어로 구성된다.
    키를 코드에 직접 하드코딩하지 않고 별도 파일에서 읽어 보안 사고를 예방한다.
    """
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)
    if not match:
        raise EnvironmentError(f"config.md에서 OpenAI API 키를 찾을 수 없습니다: {CONFIG_PATH}")
    return match.group()


def setup_models(api_key: str):
    """
    LlamaIndex 전역 설정(Settings)에 LLM과 임베딩 모델을 등록한다.

    Settings에 한 번 등록하면 이후 모든 LlamaIndex 컴포넌트가
    별도 인자 없이 자동으로 이 모델들을 사용한다.

    - LLM(gpt-4o-mini)         : 문서 청크를 읽고 엔티티와 관계를 텍스트로 추출
    - Embedding(text-embedding-3-small) : 추출된 노드와 청크를 벡터로 변환,
                                          이후 유사도 기반 검색에 활용
    """
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
        max_retries=10,
        embed_batch_size=50,
    )


def load_documents():
    """
    data/ 폴더에서 *_parsed.txt 파일을 모두 찾아 LlamaIndex Document 객체 리스트로 반환한다.

    SimpleDirectoryReader는 각 파일을 읽어 Document(text, metadata) 형태로 감싼다.
    파일이 없으면 Phase 2 파싱이 실행되지 않은 것이므로 예외를 발생시킨다.
    """
    parsed_files = list(PARSED_DOCS_DIR.glob("*_parsed.txt"))
    if not parsed_files:
        raise FileNotFoundError(
            f"*_parsed.txt 파일이 {PARSED_DOCS_DIR}에 없습니다. Phase 2 파싱을 먼저 실행하세요."
        )
    print(f"[로드] {[f.name for f in parsed_files]}")
    reader = SimpleDirectoryReader(input_files=[str(f) for f in parsed_files])
    return reader.load_data()


def build_index(documents):
    graph_store = SimplePropertyGraphStore()
    extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=20,
        num_workers=4,
    )
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )
    return index

def save_index(index):
    json_str = index.property_graph_store.graph.model_dump_json()
    KG_STORE_PATH.write_text(json_str, encoding="utf-8")
    print(f"[저장] {KG_STORE_PATH}")

def _manual_embed_nodes(index, api_key):
    import time
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    nodes = [node for nid, node in index.property_graph_store.graph.nodes.items()]
    print(f"수동 임베딩 시작: 총 {len(nodes)}개 노드")
    
    batch_size = 500
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        texts = []
        for n in batch:
            val = (n.properties or {}).get("name", n.id)
            if not val or str(val).strip() == "":
                val = "unknown"
            texts.append(str(val))
            
        print(f"  - 임베딩 중: {i}/{len(nodes)} ...")
        embeddings = embed_model.get_text_embedding_batch(texts)
        for n, emb in zip(batch, embeddings):
            n.embedding = emb
        time.sleep(2)  # Rate Limit 방지를 위한 지연

def main():
    api_key = load_api_key()
    print("=== Phase 3: KG 생성 시작 ===")
    
    # 임베딩 자동 실행을 막기 위해 embed_model을 None으로 둠
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
    Settings.embed_model = None

    print("[1/3] 문서 로드 중...")
    documents = load_documents()
    print(f"      {len(documents)}개 문서 로드 완료")

    print("[2/3] PropertyGraphIndex 생성 중... (시간이 걸릴 수 있습니다)")
    index = build_index(documents)

    print("[2.3/3] 임시 KG 저장 중 (임베딩 전)...")
    save_index(index)

    print("[2.5/3] 추출된 노드 수동 임베딩 중...")
    try:
        _manual_embed_nodes(index, api_key)
    except Exception as e:
        print(f"임베딩 중 오류 발생: {e}")

    print("[3/3] 최종 KG 저장 중...")
    save_index(index)

    print("\n=== KG 생성 완료 ===")
    print(f"저장 위치: {KG_STORE_PATH}")

if __name__ == "__main__":
    main()
