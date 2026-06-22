"""
Phase 3b — 온톨로지 기반 KG 생성 (SchemaLLMPathExtractor)

[기존 build_kg.py와의 차이]
  - 노드 타입 12종 / 엣지 타입 18종으로 추출 범위 제한 (Pydantic Literal 강제)
  - 동의어 → Canonical Name 자동 통일 (프롬프트 주입)
  - 조건부 엣지에 조건_요약·source_doc 속성 필수화 (프롬프트 주입)
  - Method C (ApplicationCondition) 노드를 1차 추출에서 직접 생성

[입력]  data/*_parsed.txt   (Phase 2에서 PDF를 파싱한 텍스트)
[출력]  data/kg_store_ontology.json
        data/kg_graph_ontology.html
"""

import argparse
import asyncio
import re
import time
from pathlib import Path
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    PropertyGraphIndex,
    Settings,
    PromptTemplate,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent.parent / "data"
CONFIG_PATH     = Path(__file__).parent.parent.parent / "config.md"
PARSED_DOCS_DIR = DATA_DIR / "parsed_docs"
KG_STORE_DIR    = DATA_DIR / "kg_store"
KG_VISUALS_DIR  = DATA_DIR / "kg_visuals"

# ── 온톨로지 스키마 ────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "LoanProduct",           # 대출 상품 (보금자리론, 디딤돌대출)
    "BorrowerRequirement",   # 채무자 자격 조건 집합
    "HouseholdType",         # 가구 유형 (신혼가구, 다자녀가구 등)
    "ApplicantType",         # 신청자 유형 (실수요자, 생애최초주택구입자 등)
    "LoanTerms",             # 대출 조건 (한도·만기·DTI/LTV)
    "PreferentialRate",      # 우대금리 항목
    "CollateralProperty",    # 담보주택
    "LoanPurpose",           # 자금용도 (구입·보전·상환)
    "Borrower",              # 채무자 (신청인)
    "Income",                # 소득 정보
    "Debt",                  # 부채 정보
    "ApplicationCondition",  # 복합 AND/OR 조건 노드 (Method C)
]

RELATION_TYPES = [
    "신청가능",       # ApplicantType/HouseholdType → LoanProduct
    "요건적용",       # LoanProduct → BorrowerRequirement
    "대출조건적용",   # LoanProduct → LoanTerms
    "소득상한_예외",  # HouseholdType → BorrowerRequirement
    "우대금리적용",   # HouseholdType/ApplicantType → PreferentialRate
    "한도상향",       # ApplicantType/HouseholdType → LoanTerms
    "DTI차감예외",    # ApplicantType → LoanTerms
    "LTV적용",        # ApplicantType → LoanTerms
    "만기조건",       # LoanTerms → BorrowerRequirement
    "담보물",         # LoanProduct → CollateralProperty
    "자금용도",       # LoanProduct → LoanPurpose
    "조건_AND",       # ApplicationCondition → ApplicantType/HouseholdType
    "조건_OR",        # ApplicationCondition → ApplicantType/HouseholdType
    "결과적용",       # ApplicationCondition → LoanTerms/PreferentialRate
    "구성가구",       # Borrower → HouseholdType
    "해당유형",       # Borrower → ApplicantType
    "보유소득",       # Borrower → Income
    "보유부채",       # Borrower → Debt
    # ── 아래는 \x1c 오염을 유발하던 미분류 관계를 흡수하기 위해 추가 ──
    "나열",           # 열거·순서 관계 (구입→보전→상환 등 목록 나열)
    "포함",           # 상위→하위 포함 관계 (소득입증자료→근로소득 등)
    "관련",           # 위 범주에 맞지 않는 일반 연관 관계 (fallback)
]

# (출발 노드 타입, 관계, 도착 노드 타입) — strict=False이므로 참고용
VALIDATION_SCHEMA = [
    ("HouseholdType",        "신청가능",       "LoanProduct"),
    ("ApplicantType",        "신청가능",       "LoanProduct"),
    ("LoanProduct",          "요건적용",       "BorrowerRequirement"),
    ("LoanProduct",          "대출조건적용",   "LoanTerms"),
    ("HouseholdType",        "소득상한_예외",  "BorrowerRequirement"),
    ("HouseholdType",        "우대금리적용",   "PreferentialRate"),
    ("ApplicantType",        "우대금리적용",   "PreferentialRate"),
    ("ApplicantType",        "한도상향",       "LoanTerms"),
    ("HouseholdType",        "한도상향",       "LoanTerms"),
    ("ApplicantType",        "DTI차감예외",    "LoanTerms"),
    ("ApplicantType",        "LTV적용",        "LoanTerms"),
    ("LoanTerms",            "만기조건",       "BorrowerRequirement"),
    ("LoanProduct",          "담보물",         "CollateralProperty"),
    ("LoanProduct",          "자금용도",       "LoanPurpose"),
    ("ApplicationCondition", "조건_AND",       "ApplicantType"),
    ("ApplicationCondition", "조건_AND",       "HouseholdType"),
    ("ApplicationCondition", "조건_OR",        "ApplicantType"),
    ("ApplicationCondition", "조건_OR",        "HouseholdType"),
    ("ApplicationCondition", "결과적용",       "LoanTerms"),
    ("ApplicationCondition", "결과적용",       "PreferentialRate"),
    ("Borrower",             "구성가구",       "HouseholdType"),
    ("Borrower",             "해당유형",       "ApplicantType"),
    ("Borrower",             "보유소득",       "Income"),
    ("Borrower",             "보유부채",       "Debt"),
]

# Canonical Name 매핑 (동의어 → 표준명)
CANONICAL_MAP = {
    "주금공": "주택금융공사",
    "HF": "주택금융공사",
    "공사": "주택금융공사",
    "한국주택금융공사": "주택금융공사",
    "보금자리 대출": "보금자리론",
    "u-보금자리론": "보금자리론",
    "보금자리": "보금자리론",
    "내집마련디딤돌": "디딤돌대출",
    "디딤돌론": "디딤돌대출",
    "디딤돌 대출": "디딤돌대출",
    "신혼부부": "신혼가구",
    "혼인 7년 이내 가구": "신혼가구",
    "생애최초": "생애최초주택구입자",
    "최초 구입자": "생애최초주택구입자",
    "생애최초 주택구입": "생애최초주택구입자",
    "실거주자": "실수요자",
    "실수요": "실수요자",
    "전세사기 피해자": "전세사기피해자등",
    "전세사기 피해": "전세사기피해자등",
}

# 사전 정의 ApplicationCondition ID (docs/ontology_design.md 8-2 참조)
CONDITION_IDS = [
    ("cond_AND_생애최초_실수요자_LTV80",
     "생애최초주택구입자 AND 실수요자 → LTV 80% (보금자리론)"),
    ("cond_AND_생애최초_실수요자_한도상향",
     "생애최초주택구입자 AND 실수요자 → 한도 상향 (보금자리론)"),
    ("cond_OR_실수요자_생애최초_전세사기_DTI차감예외",
     "실수요자 OR 생애최초주택구입자 OR 전세사기피해자등 → DTI 차감 예외"),
]

# 엣지에 힌트로 제공할 속성 목록 — possible_relation_props에 전달
RELATION_PROPS = [
    ("조건_요약",      "조건부 엣지의 인간 가독 조건 서술. 예: '신혼가구, 연소득 8,500만 이하'"),
    ("source_doc",     "규정 조항 출처. 예: '보금자리론 업무처리기준 §12.3'"),
    ("소득_상한",      "연소득 상한 (정수, 원 단위). 예: 85000000"),
    ("나이_상한",      "나이 상한 (정수, 세). 예: 40"),
    ("용도_조건",      "자금용도 조건: '구입' | '보전' | '상환'"),
    ("상품",           "적용 상품: '보금자리론' | '디딤돌대출' | '공통'"),
    ("주택가격_상한",  "담보주택 가격 상한 (정수, 원 단위). 예: 600000000"),
    ("우대율",         "우대금리 (소수). 예: 0.003 (= 0.3%p)"),
    ("LTV상한",        "LTV 상한 (소수). 예: 0.80"),
    ("한도",           "대출 한도 (정수, 원 단위). 예: 400000000"),
    ("만기",           "대출 만기 (정수, 년). 예: 50"),
    ("차감율",         "DTI 차감률 (소수). 예: 0.10"),
]


def load_api_key() -> str:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)
    if not match:
        raise EnvironmentError(f"config.md에서 OpenAI API 키를 찾을 수 없습니다: {CONFIG_PATH}")
    return match.group()


def setup_models(api_key: str):
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=api_key, timeout=600.0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)


def load_documents(test: bool = False):
    parsed_files = list(PARSED_DOCS_DIR.glob("*_parsed.txt"))
    if not parsed_files:
        raise FileNotFoundError(
            f"*_parsed.txt 파일이 {PARSED_DOCS_DIR}에 없습니다. Phase 2 파싱을 먼저 실행하세요."
        )
    if test:
        parsed_files = parsed_files[:1]
    print(f"[로드] {[f.name for f in parsed_files]}")
    reader = SimpleDirectoryReader(input_files=[str(f) for f in parsed_files])
    docs = reader.load_data()
    if test:
        # 첫 번째 청크만, 텍스트는 2000자로 자름
        truncated = docs[0].get_content()[:2000]
        docs = [Document(text=truncated)]
        print(f"[테스트 모드] 문서 1개, 텍스트 {len(truncated)}자로 제한")
    return docs


def build_extraction_prompt() -> PromptTemplate:
    """
    SchemaLLMPathExtractor에 전달할 커스텀 추출 프롬프트.

    사용 가능한 템플릿 변수:
      {text}                   — 추출 대상 텍스트 청크 (자동 주입)
      {max_triplets_per_chunk} — 최대 트리플 수 (자동 주입)
    스키마(노드·엣지 타입)는 Pydantic Literal 모델로 별도 강제된다.
    """
    canonical_lines = "\n".join(
        f"  {src} → {dst}" for src, dst in CANONICAL_MAP.items()
    )
    condition_lines = "\n".join(
        f"  {cid}: {desc}" for cid, desc in CONDITION_IDS
    )

    template = f"""\
당신은 주택금융 규정 문서에서 지식 그래프 트리플을 추출하는 전문가입니다.
추출할 수 있는 트리플은 최대 {{max_triplets_per_chunk}}개입니다.

=== [절대 규칙] 출력 금지 사항 ===
- 관계(relation) 레이블에 ASCII 제어문자(코드 0~31)를 절대 포함하지 말 것.
  특히 \\x1c, \\x1e, \\x00 등의 문자는 사용 금지. 위반 시 해당 트리플은 폐기된다.
- 노드 이름에 제어문자를 포함하지 말 것.
- 관계 레이블은 반드시 제공된 목록(신청가능·요건적용·나열·포함·관련 등) 중 하나를 사용할 것.
  목록에 맞는 것이 없으면 "관련"을 사용하라. 임의 문자를 만들지 말 것.

=== Canonical Name 규칙 (반드시 준수) ===
아래 동의어는 반드시 표준명으로 변환하라. 원문 표현을 그대로 노드명으로 쓰지 말 것.
{canonical_lines}

=== [핵심] 노드 이름에 실제 값을 담아라 ===
목적어(object) 노드 이름에는 가능한 한 원문의 구체적인 수치·조건을 포함하라.
추상적인 레이블만 있는 노드보다 실제 값이 담긴 노드가 검색 품질을 높인다.

  [Bad]  (PreferentialRate:다자녀가구_우대금리_2019.12.31이후)   ← 수치 없음
  [Good] (PreferentialRate:다자녀가구_우대금리_0.7%p차감)        ← 수치 포함

  [Bad]  (LoanTerms:보금자리론_한도상향)                        ← 수치 없음
  [Good] (LoanTerms:보금자리론_한도_4억원)                       ← 수치 포함

  [Bad]  (BorrowerRequirement:만기조건_나이)                    ← 수치 없음
  [Good] (BorrowerRequirement:50년_만기_만35세_미만)             ← 수치 포함

=== 엣지 속성 규칙 (Method A) ===
조건부 의미를 가진 엣지(우대금리적용·소득상한_예외·한도상향·DTI차감예외·LTV적용·만기조건)에는
반드시 다음 속성을 포함하라:
  - 조건_요약: 사람이 읽을 수 있는 조건 서술 (예: "신혼가구, 연소득 8,500만 이하")

해당하는 경우 추가 속성도 포함하라:
  소득_상한(원), 나이_상한(세), 용도_조건("구입"|"보전"|"상환"),
  상품("보금자리론"|"디딤돌대출"|"공통"), 주택가격_상한(원),
  우대율(소수, 예:0.003), LTV상한(소수), 한도(원), 만기(년), 차감율(소수)

※ source_doc 속성은 원문에서 조항 번호를 확인할 수 있을 때만 기재하라.
  불명확하면 source_doc를 아예 생략하라. "§X.Y" 같은 플레이스홀더는 쓰지 말 것.

=== Method C 규칙 (AND/OR 복합 조건) ===
텍스트에서 두 개 이상의 신청자유형(ApplicantType) 또는 가구유형(HouseholdType)을
동시에(AND) 또는 선택적으로(OR) 만족해야 하는 조건이 나오면:

1. ApplicationCondition 중간 노드를 먼저 생성하라.
2. 아래 사전 정의 ID를 우선 사용하라:
{condition_lines}
3. 목록에 없는 새 조건은 형식 cond_{{AND|OR}}_{{요소1}}_{{요소2}}..._{{결과}} 로 만들어라.
   예: cond_AND_신혼가구_생애최초_우대금리
4. ApplicationCondition 노드의 조건_요약 속성에는 실제 조건 내용을 자연어로 기술하라.
   예: "신혼가구이면서 생애최초 주택구입자인 경우 LTV 80% 적용"

Method C 트리플 예시 (생애최초 AND 실수요자 → LTV 80%):
  (ApplicationCondition:cond_AND_생애최초_실수요자_LTV80) -[조건_AND]→ (ApplicantType:생애최초주택구입자)
  (ApplicationCondition:cond_AND_생애최초_실수요자_LTV80) -[조건_AND]→ (ApplicantType:실수요자)
  (ApplicationCondition:cond_AND_생애최초_실수요자_LTV80) -[결과적용 {{LTV상한:0.80, 조건_요약:"생애최초주택구입자 AND 실수요자 시 LTV 80% 적용"}}]→ (LoanTerms:보금자리론_LTV_80%)

=== 관계 타입 선택 가이드 ===
- 항목을 열거할 때 (A, B, C 나열): "나열" 사용
- 상위 개념이 하위 항목을 포함할 때: "포함" 사용
- 위 모든 관계에 해당하지 않는 연관: "관련" 사용
- 관계를 알 수 없거나 제어문자를 쓰고 싶다면: "관련" 사용

=== 추출 예시 ===
[Good] 신혼가구가 보금자리론에서 우대금리를 받는 경우:
  (HouseholdType:신혼가구) -[우대금리적용 {{우대율:0.002, 소득_상한:85000000, 용도_조건:"구입", 조건_요약:"신혼가구, 연소득 8,500만 이하, 구입용도", 상품:"보금자리론"}}]→ (PreferentialRate:신혼가구_보금자리론_우대금리_0.2%p)

[Good] 다자녀가구 디딤돌대출 우대금리:
  (HouseholdType:다자녀가구) -[우대금리적용 {{우대율:0.007, 조건_요약:"자녀 3명 이상 다자녀가구"}}]→ (PreferentialRate:다자녀가구_우대금리_0.7%p차감)

[Good] 자금용도 열거:
  (LoanPurpose:구입용도) -[나열]→ (LoanPurpose:보전용도)
  (LoanPurpose:보전용도) -[나열]→ (LoanPurpose:상환용도)

[Bad — 아래처럼 추출하지 말 것]
  (신혼부부) -[관계]→ (우대금리)                    ← 동의어, 타입 없음, 속성 없음
  (보금자리론) -[\\x1c\\x1c\\x1c]→ (이용 불가)     ← 제어문자 절대 금지

=== 추출 대상 텍스트 ===
{{text}}

=== 출력 지침 ===
- 관계 레이블: 제공된 목록 중 하나만 사용. 없으면 "관련". 제어문자 절대 금지.
- 노드 이름: 가능하면 구체적인 수치나 조건을 포함시켜라.
- 조건부 엣지: 조건_요약 속성을 반드시 포함하라.
- ApplicationCondition 노드: 조건_요약에 자연어 설명을 포함하라.
"""
    return PromptTemplate(template)


class RateLimitedSchemaExtractor(SchemaLLMPathExtractor):
    """
    SchemaLLMPathExtractor에 토큰 버킷 레이트 리미터를 추가한 서브클래스.

    acall()만 override하므로 Pydantic 필드 충돌 없음.
    요청 dispatch를 min_interval(=60/rpm 초) 간격으로 throttle하여
    num_workers개 동시 in-flight를 유지하면서 RPM 상한을 지킨다.
    """

    def __init__(self, *args, rpm: int = 400, **kwargs):
        super().__init__(*args, **kwargs)
        # Pydantic __setattr__을 우회해 인스턴스에 직접 저장
        object.__setattr__(self, "_rpm", rpm)

    async def acall(self, nodes, show_progress: bool = False, **_):
        from llama_index.core.async_utils import run_jobs

        interval = 60.0 / self._rpm
        lock  = asyncio.Lock()
        state = {"last": 0.0}

        async def throttled(node):
            async with lock:
                now  = time.monotonic()
                wait = state["last"] + interval - now
                if wait > 0:
                    await asyncio.sleep(wait)
                state["last"] = time.monotonic()
            return await self._aextract(node)

        jobs = [throttled(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text with schema",
        )


def _has_control_chars(s: str) -> bool:
    return any(ord(c) < 32 for c in s)


def _postprocess_graph(graph_store: SimplePropertyGraphStore) -> tuple[int, int]:
    """
    추출 후 그래프를 정제한다.

    1. 관계 레이블에 제어문자(\x00-\x1f)가 포함된 triplet 제거
    2. cond_ 노드: 연결된 조건·결과 엣지를 탐색해 자연어 설명 구성
       → context로 반환될 때 LLM이 조건 내용을 실제로 활용 가능하게 함
    3. 일반 entity 노드: 'name' 텍스트 프로퍼티 주입
    """
    # ── 1. 오염 triplet 제거 ──────────────────────────────────────────────────
    removed = 0
    bad_rel_ids = [
        rid for rid, rel in graph_store.graph.relations.items()
        if _has_control_chars(rel.label)
    ]
    for rid in bad_rel_ids:
        graph_store.graph.relations.pop(rid, None)
        removed += 1

    graph_store.graph.triplets = [
        t for t in graph_store.graph.triplets
        if not _has_control_chars(str(t[1]) if len(t) > 1 else "")
    ]

    # ── 2. cond_ 노드의 outgoing 엣지 수집 ───────────────────────────────────
    cond_edges: dict[str, dict] = {}
    for rel in graph_store.graph.relations.values():
        if not rel.source_id.startswith("cond_"):
            continue
        entry = cond_edges.setdefault(
            rel.source_id, {"and": [], "or": [], "result": []}
        )
        rel_props = rel.properties or {}
        if rel.label == "조건_AND":
            entry["and"].append(rel.target_id)
        elif rel.label == "조건_OR":
            entry["or"].append(rel.target_id)
        elif rel.label == "결과적용":
            result_text = rel.target_id
            # 엣지 속성에서 구체적인 수치 보강
            if "조건_요약" in rel_props:
                result_text += f" ({rel_props['조건_요약']})"
            elif "LTV상한" in rel_props:
                result_text += f" (LTV {float(rel_props['LTV상한'])*100:.0f}%)"
            elif "한도" in rel_props:
                result_text += f" (한도 {int(rel_props['한도']):,}원)"
            elif "우대율" in rel_props:
                result_text += f" (우대 {float(rel_props['우대율'])*100:.2f}%p)"
            entry["result"].append(result_text)

    # ── 3. 노드 텍스트 주입 ───────────────────────────────────────────────────
    injected = 0
    for nid, node in graph_store.graph.nodes.items():
        if node.label == "text_chunk":
            continue
        props = node.properties or {}

        if nid.startswith("cond_"):
            # 기존 조건_요약이 충분히 풍부하면 그대로 활용
            existing = props.get("조건_요약", "")
            edges = cond_edges.get(nid, {})
            if len(existing) > 15:
                rich = existing
            else:
                and_parts = edges.get("and", [])
                or_parts  = edges.get("or",  [])
                result_parts = edges.get("result", [])
                cond_str = ""
                if and_parts:
                    cond_str = "(" + " AND ".join(and_parts) + ")"
                elif or_parts:
                    cond_str = "(" + " OR ".join(or_parts) + ")"
                result_str = (" → " + ", ".join(result_parts)) if result_parts else ""
                rich = (cond_str + result_str) if (cond_str or result_str) else nid
            props["name"] = f"[복합조건] {rich}"
            props["조건_요약"] = rich
        else:
            if "name" not in props:
                props["name"] = nid

        node.properties = props
        injected += 1

    return removed, injected


def _re_embed_nodes(graph_store: SimplePropertyGraphStore) -> int:
    """
    _postprocess_graph()로 name이 변경된 entity 노드의 embedding을 재생성한다.

    from_documents() 시 embedding은 원본 노드 ID 텍스트 기준으로 생성된다.
    postprocessing이 name을 풍부한 자연어로 교체하면 embedding과 name이 불일치하여
    vector search 품질이 저하된다. 이 함수는 최종 name 기준으로 재생성해 일치시킨다.
    """
    entity_nodes = [
        (nid, node)
        for nid, node in graph_store.graph.nodes.items()
        if node.label != "text_chunk"
    ]
    if not entity_nodes:
        return 0

    texts = [
        (node.properties or {}).get("name", nid)
        for nid, node in entity_nodes
    ]
    embeddings = Settings.embed_model.get_text_embedding_batch(texts, show_progress=True)

    for (_, node), embedding in zip(entity_nodes, embeddings):
        node.embedding = embedding

    return len(entity_nodes)


def build_ontology_index(documents, test: bool = False):
    """
    RateLimitedSchemaExtractor로 온톨로지 정합 PropertyGraphIndex를 생성한다.

    - 실제 run: workers=100, rpm=400 → 최대 100개 동시 in-flight, 400 RPM throttle
    - 테스트 run: workers=4, rpm=400 (청크 수가 적어 실질적으로 영향 없음)
    """
    extractor = RateLimitedSchemaExtractor(
        llm=Settings.llm,
        extract_prompt=build_extraction_prompt(),
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        possible_relation_props=RELATION_PROPS,
        kg_validation_schema=VALIDATION_SCHEMA,
        strict=False,
        max_triplets_per_chunk=5 if test else 20,
        num_workers=4 if test else 8,
        allow_additional_properties=False,  # OpenAI structured output 요구사항
        rpm=70,  # TPM 200K ÷ ~2500 tokens/req ≈ 80 req/min → 안전 마진 포함 70
    )
    graph_store = SimplePropertyGraphStore()
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    print("[후처리] 오염 triplet 제거 및 entity 노드 텍스트 주입 중...")
    removed, injected = _postprocess_graph(graph_store)
    print(f"         제거된 오염 triplet: {removed}개 | name 주입된 노드: {injected}개")

    print("[후처리] entity 노드 임베딩 재생성 중... (name 변경 후 embedding 불일치 수정)")
    reembedded = _re_embed_nodes(graph_store)
    print(f"         재임베딩 완료: {reembedded}개 노드")

    return index



def main():
    parser = argparse.ArgumentParser(description="Phase 3b: 온톨로지 기반 KG 생성")
    parser.add_argument(
        "--test",
        action="store_true",
        help="테스트 모드: 문서 1개·텍스트 2000자·트리플 5개로 제한, 별도 파일에 저장",
    )
    args = parser.parse_args()

    kg_store_path = KG_STORE_DIR  / ("kg_store_ontology_test.json" if args.test else "kg_store_ontology.json")
    html_path     = KG_VISUALS_DIR / ("kg_graph_ontology_test.html" if args.test else "kg_graph_ontology.html")

    api_key = load_api_key()
    mode_label = "[테스트 모드]" if args.test else ""
    print(f"=== Phase 3b: 온톨로지 기반 KG 생성 시작 {mode_label} ===")

    setup_models(api_key)

    print("[1/3] 문서 로드 중...")
    documents = load_documents(test=args.test)
    print(f"      {len(documents)}개 문서 로드 완료")

    print("[2/3] SchemaLLMPathExtractor로 KG 생성 중... (시간이 걸릴 수 있습니다)")
    index = build_ontology_index(documents, test=args.test)

    print("[3/3] KG 저장 중...")
    json_str = index.property_graph_store.graph.model_dump_json()
    kg_store_path.write_text(json_str, encoding="utf-8")
    print(f"[저장] {kg_store_path}")

    # 시각화 — visualize_kg.py의 main()에 경로 인자 전달
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from visualize_kg import main as visualize_main
        print("[시각화] 그래프 HTML 생성 중...")
        visualize_main(kg_path=kg_store_path, output_path=html_path)
    except Exception as e:
        print(f"[시각화 건너뜀] {e}")

    print("\n=== 온톨로지 기반 KG 생성 완료 ===")
    print(f"KG 저장 위치: {kg_store_path}")
    print(f"시각화 위치 : {html_path}")


if __name__ == "__main__":
    main()
