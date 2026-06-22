import os
import sys
import re
import json
import time
import asyncio
import threading
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset

os.environ["TQDM_DISABLE"] = "1"

# Python 3.12+ Windows에서 SelectorEventLoop 사용 (호환성)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# datasets + Python 3.13 Windows 멀티프로세싱 데드락 방지
from datasets import Dataset as _Dataset
_orig_map = _Dataset.map
def _map_single_proc(self, *args, **kwargs):
    kwargs.pop("num_proc", None)
    return _orig_map(self, *args, **kwargs)
_Dataset.map = _map_single_proc

# ── Ragas 0.1.7 + Python 3.13 호환성 패치 ─────────────────────────────────────
# 근본 원인:
#   Python 3.13의 _AsCompletedIterator.__init__()은 get_event_loop()로
#   태스크를 현재 스레드의 이벤트 루프에 바인딩함.
#   Ragas Runner.__init__()은 self.loop = new_event_loop()를 만든 뒤
#   as_completed()를 호출하지만, self.loop를 set_event_loop()로 등록하지 않아
#   태스크가 다른 루프에 바인딩됨.
#   Runner.run()은 self.loop로 실행하므로 태스크가 영원히 완료 신호를 못 받음 → 데드락.
# 해결:
#   Runner.__init__에서 as_completed() 호출 전 set_event_loop(self.loop)를 삽입.
import ragas.executor as _ragas_executor
from ragas.run_config import RunConfig as _RunConfig
import numpy as _np

class _FixedRunner(_ragas_executor.Runner):
    def __init__(self, jobs, desc, keep_progress_bar=True, raise_exceptions=True, run_config=None):
        threading.Thread.__init__(self)
        self.jobs = jobs
        self.desc = desc
        self.keep_progress_bar = keep_progress_bar
        self.raise_exceptions = raise_exceptions
        self.run_config = run_config or _RunConfig()
        self.loop = asyncio.new_event_loop()
        # 핵심 수정: as_completed() 전에 self.loop를 현재 루프로 등록
        asyncio.set_event_loop(self.loop)
        self.futures = _ragas_executor.as_completed(
            loop=self.loop,
            coros=[coro for coro, _ in self.jobs],
            max_workers=self.run_config.max_workers,
        )

    def run(self):
        asyncio.set_event_loop(self.loop)
        results = []
        try:
            results = self.loop.run_until_complete(self._aresults())
        finally:
            self.results = results
            self.loop.stop()

    async def _aresults(self):
        from tqdm.auto import tqdm
        from ragas.exceptions import MaxRetriesExceeded
        import logging as _logging
        _logger = _logging.getLogger("ragas.runner")

        results = []
        total = len(self.jobs)
        for i, future in enumerate(tqdm(
            self.futures,
            desc=self.desc,
            total=total,
            leave=self.keep_progress_bar,
        )):
            r = (-1, _np.nan)
            print(f"  [Ragas] {i+1}/{total} 계산 중...", flush=True)
            try:
                r = await future
                score_str = f"{r[1]:.4f}" if not _np.isnan(r[1]) else "nan"
                print(f"  [Ragas] {i+1}/{total} 완료 → score={score_str}", flush=True)
            except MaxRetriesExceeded as e:
                _logger.warning(f"max retries exceeded for {e.evolution}")
                print(f"  [Ragas] {i+1}/{total} 재시도 초과", flush=True)
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    _logger.error("Runner raised an exception", exc_info=True)
                    print(f"  [Ragas] {i+1}/{total} 오류: {e}", flush=True)
            results.append(r)
        return results

_ragas_executor.Runner = _FixedRunner
# ─────────────────────────────────────────────────────────────────────────────

from llama_index.core import Settings, PropertyGraphIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from kg_prompt import KG_QA_TEMPLATE

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    answer_correctness,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.parent.parent
DATA_DIR         = BASE_DIR / "data"
CONFIG_PATH      = BASE_DIR / "config.md"
KG_STORE_DIR     = DATA_DIR / "kg_store"
PARSED_DOCS_DIR  = DATA_DIR / "parsed_docs"
EVAL_RESULTS_DIR = DATA_DIR / "eval_results"

def load_api_key() -> str:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)
    if not match:
        raise EnvironmentError("config.md에서 API 키를 찾을 수 없습니다.")
    api_key = match.group()
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def init_models(api_key: str):
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

def load_index(kg_store_path: Path):
    if not kg_store_path.exists():
        raise FileNotFoundError(f"KG 파일이 존재하지 않습니다: {kg_store_path}")
    data = json.loads(kg_store_path.read_text(encoding="utf-8"))
    graph_store = SimplePropertyGraphStore.from_dict(data)
    return PropertyGraphIndex.from_existing(property_graph_store=graph_store)

# ── 포트폴리오용 핵심 평가 데이터셋 (Golden Dataset) ──────────────────────────
eval_dataset_raw = [
    {"question": "디딤돌 대출 신청 시 부부합산 연간 총소득 기준은 얼마인가요(일반 가구)?", "ground_truth": "대출신청인과 배우자의 합산 총소득이 연간 60백만원 이하여야 합니다."},
    {"question": "생애최초 주택구입자가 디딤돌 대출을 받을 때 담보주택 당 최대 대출한도는 얼마인가요?", "ground_truth": "생애최초 주택구입자인 경우 담보주택 당 2.4억원 이내입니다."},
    {"question": "디딤돌 대출의 다자녀가구 우대금리는 얼마인가요?", "ground_truth": "다자녀가구의 우대금리는 차감 0.7%p 입니다."},
    {"question": "디딤돌 대출 상환방식 중 체증식 분할상환을 선택할 수 있는 조건은 무엇인가요?", "ground_truth": "접수일 현재 만 40세 미만 근로자이고 고정금리를 선택한 경우에만 허용됩니다."},
    {"question": "디딤돌 대출에서 대출 실행일로부터 3년 이내 원금 상환 시 중도상환수수료율 최고한도는 얼마인가요?", "ground_truth": "대출실행일로부터 경과일수별로 0.6% 한도 내에서 부과됩니다."},
    {"question": "현재 다른 목적물의 주택담보대출을 이용 중인 경우 디딤돌 대출 취급이 가능한가요?", "ground_truth": "불가능합니다. 한국신용정보원 조회 결과 주택담보대출을 이용 중이면 대출불가입니다."},
    {"question": "디딤돌 대출 심사 시 기본 DTI(총부채상환비율) 한도 기준은 몇 %인가요?", "ground_truth": "60% 이내입니다."},
    {"question": "주택을 상속받아 취득한 지 1개월 지났을 경우, 디딤돌 대출을 구입용도로 받을 수 있나요?", "ground_truth": "상속으로 주택을 취득한 경우 등기일로부터 3개월 이내라도 구입용도 취급이 불가능합니다."},
    {"question": "만 30세 이상 미혼 단독세대주의 디딤돌 대출 기본 한도는 얼마인가요?", "ground_truth": "일반의 경우 1.5억원 이내입니다."},
    {"question": "디딤돌 대출 우대금리를 전부 적용한 후, 최종 대출금리의 하한선(최저금리)은 얼마인가요?", "ground_truth": "최종 대출금리의 하한선은 연 1.5%입니다 (단, 생애최초 신혼가구는 연 1.2%)."},
    {"question": "보금자리론의 다자녀가구 최대 대출 한도는 얼마인가요?", "ground_truth": "다자녀가구의 경우 보금자리론 담보주택 당 4억원 이내입니다."},
    {"question": "보금자리론 50년 만기 대출을 신청할 수 있는 연령 조건(일반 가구 기준)은 어떻게 되나요?", "ground_truth": "만 35세 미만의 채무자만 50년 만기를 신청할 수 있습니다."},
    {"question": "보금자리론을 받을 담보주택이 민간임대주택에 관한 특별법에 따라 등록된 임대주택인 경우 대출 가능한가요?", "ground_truth": "등록된 임대주택은 보금자리론 취급 불가입니다."},
    {"question": "보금자리론 신혼가구의 부부 합산 연소득 제한은 얼마인가요?", "ground_truth": "신혼가구의 부부 합산 연소득은 85백만원 이하여야 합니다."},
    {"question": "보금자리론 취급을 위한 최소 NICE신용평가점수(CB점수)는 얼마입니까?", "ground_truth": "NICE신용평가(주) 기준 CB점수가 271점 이상이어야 취급 가능합니다."},
    {"question": "아낌e-보금자리론(전자약정 및 전자등기) 이용 시 우대금리는 얼마인가요?", "ground_truth": "0.1%p의 우대금리가 적용됩니다."},
    {"question": "보금자리론에서 조기상환수수료율은 최대 몇 % 이내에서 부과되나요?", "ground_truth": "최초 실행일로부터 3년 이내 남은 일수에 따라 최대 0.5% 한도 내에서 부과됩니다."},
    {"question": "보금자리론에서 담보주택이 규제지역에 있는 경우 기본 DTI에서 어떻게 차감하나요?", "ground_truth": "규제지역인 경우 기본 DTI에서 10%p를 차감하여 적용합니다 (즉, 50% 적용)."},
    {"question": "보금자리론 자금용도가 구입용도와 보전용도가 중복되는 경우 우선순위는 어떻게 되나요?", "ground_truth": "자금용도가 중복될 경우 구입용도, 보전용도, 상환용도 순으로 순차적용합니다."},
    {"question": "대체취득(일시적 2주택) 목적으로 보금자리론 구입용도 대출을 받을 때, 기존주택 처분 기한은 언제까지인가요?", "ground_truth": "기존 주택의 처분기한은 대출 실행일로부터 3년입니다."}
]

eval_dataset_hard = [
    {"question": "저는 만 31세 미혼 단독세대주로 현재 디딤돌 대출을 받아 소형 주택에 거주 중입니다. 다음 달에 결혼을 앞두고 있어(청첩장 보유), 신혼가구 자격으로 새로운 4억 원짜리 아파트를 구입하려고 합니다. 기존 주택을 처분하는 조건으로 내집마련 디딤돌 대출을 새로 받을 수 있는지, 만약 가능하다면 최대한도는 얼마인가요?", "ground_truth": "가능하며, 최대한도는 3.2억 원입니다. 만 30세 이상 미혼 단독세대주 조건으로 이미 대출을 이용 중이라도, 결혼예정자로서 신혼가구 자격을 충족하고 기존 주택을 처분하는 조건이면 대출이 가능합니다. 신혼가구의 담보주택 당 최대 한도는 3.2억 원입니다."},
    {"question": "보금자리론을 이용해 아파트를 구입하여 거주 중인 1주택자입니다. 최근 새로운 아파트의 분양권을 취득하게 되었는데, 기존 보금자리론의 대출 회수(기한이익상실)를 막기 위해서 언제까지 추가로 취득한 분양권을 처분해야 하나요?", "ground_truth": "검증기준일로부터 3년 이내입니다. 일반 추가주택은 검증기준일로부터 6개월 내 처분해야 하나, 분양권이나 조합원 입주권으로 취득한 예외적인 경우에는 검증기준일로부터 3년의 처분기한이 부여됩니다."},
    {"question": "연소득이 6,800만 원인 부부입니다. 자녀는 없으며 이번에 처음으로 5억 원짜리 아파트를 구입하려고 합니다. 내집마련 디딤돌 대출을 신청할 수 있는지, 가능하다면 적용받을 수 있는 최대 한도와 기본 우대금리 항목은 무엇인가요?", "ground_truth": "신청 가능하며, 최대 한도는 2.4억 원, 0.2%p 생애최초 우대금리가 적용됩니다. 일반 가구의 소득 한도는 6천만 원이나, 생애최초 주택구입자는 7천만 원 이하까지 허용됩니다. 생애최초 구입자의 최대 한도는 2.4억 원이며, 해당 자격으로 0.2%p의 우대금리를 받습니다."},
    {"question": "연소득 6,500만 원인 무주택 부부가 규제지역(조정대상지역)에 있는 5억 5천만 원짜리 아파트를 구입하기 위해 보금자리론을 신청하려고 합니다. 이 경우 적용되는 DTI(총부채상환비율) 최대한도는 몇 %인가요?", "ground_truth": "60%입니다. 규제지역은 DTI에서 10%p를 차감해야 하나, 부부합산 연소득 7천만 원 이하, 담보주택 가격 6억 원 이하, 무주택 구입용도라는 실수요자 요건을 충족하므로 차감 없이 기본 60%가 적용됩니다."},
    {"question": "「전세사기피해자 지원 및 주거안정에 관한 특별법」에 따른 전세사기피해자(연소득 9,000만 원)가 주택을 구입하기 위해 보금자리론을 신청하려 합니다. 신청 가능 여부와 대출 한도, 적용 우대금리는 얼마인가요?", "ground_truth": "신청 가능하며, 최대 4억 원 한도에 1.0%p 우대금리를 받습니다. 전세사기피해자는 7천만원 소득 제한이 없으며, 최대한도는 4억 원이고 전용 우대금리 1.0%p가 적용됩니다."},
    {"question": "부모님과 떨어져 만 17세인 남동생과 함께 8개월째 동일 세대를 구성해 살고 있는 28세 미혼 세대주입니다. 내집마련 디딤돌 대출 신청이 가능한지, 가능하다면 대출 최대한도는 얼마인가요?", "ground_truth": "신청 가능하며, 최대한도는 1.5억 원입니다. 만 30세 미만 단독세대주는 취급 불가이나, 미성년 형제·자매 1인 이상과 동일세대를 구성하고 부양기간이 6개월 이상이므로 세대주로 인정됩니다. 미혼이므로 한도는 1.5억 원입니다."},
    {"question": "저희 부부는 현재 모두 38세이며, 혼인신고를 한 지 5년 된 신혼가구입니다. 보금자리론을 통해 '50년 만기'에 '체증식 분할상환' 방식으로 대출 신청이 가능한가요?", "ground_truth": "불가합니다. 신혼가구는 만 40세 미만까지 50년 만기를 신청할 수 있어 가능하나, 체증식 분할상환 방식은 50년 만기 상품에 적용할 수 없도록 규정되어 있습니다."},
    {"question": "디딤돌 대출을 받아 거주하던 중, 시골 주택을 단독 상속받았습니다. 추가 주택 소유로 인해 대출이 회수(기한의 이익 상실)되는 것을 막으려면 상속 주택을 언제까지 처분해야 하나요?", "ground_truth": "국토교통부 회신일로부터 6개월 이내입니다. 대출 실행 후 단독 상속으로 취득한 주택은 사후관리 시 무주택으로 보는 예외 규정에 해당하여, 6개월 이내에 처분하면 기한의 이익 상실을 면할 수 있습니다."},
    {"question": "부부합산 연소득이 6,500만 원이며 남편은 34세, 아내는 33세인 부부가 6개월 전 자녀를 출산했습니다. 아낌e-보금자리론 이용 시, 중복 적용 가능한 우대금리 항목과 총합은 몇 %p인가요?", "ground_truth": "총 0.4%p 우대금리를 적용받습니다. 아낌e(0.1%), 저소득청년(0.1%), 신생아출산가구(0.2%) 요건을 모두 충족하며, 규정상 이 항목들은 중복 적용이 허용됩니다."},
    {"question": "디딤돌 대출을 연 5.5%로 이용 중 4개월째 연체하고 있습니다. 기한의 이익 상실 전일 때, 납입 지연된 원리금에 부과되는 지연배상금(연체이자)률은 연 몇 %인가요?", "ground_truth": "연 10%입니다. 연체 3개월 초과 시 원칙적인 지연배상금률은 10.5%(5.5+5)이나, 디딤돌 규정상 지연배상금률이 연 10%를 초과할 경우 최고 한도인 연 10%까지만 적용하도록 제한하고 있습니다."}
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="빠른 테스트 실행 (3문항, top_k=1)")
    parser.add_argument(
        "--ontology",
        action="store_true",
        help="온톨로지 기반 KG 사용 (kg_store_ontology.json → eval_results_ontology.csv)",
    )
    parser.add_argument(
        "--naive",
        action="store_true",
        help="KG 없이 일반 Naive RAG 사용 (텍스트 문서 → eval_results_naive.csv)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Naive RAG와 온톨로지 KG를 앙상블(RRF)하여 사용 (eval_results_ensemble.csv)",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="단순 질문 대신 Multi-hop 구조의 Hard Dataset 10문항으로 평가를 수행합니다.",
    )
    args = parser.parse_args()

    is_test = args.test
    use_ontology = args.ontology
    use_naive = args.naive
    use_ensemble = args.ensemble
    
    target_dataset = eval_dataset_hard if args.hard else eval_dataset_raw
    dataset = target_dataset[:3] if is_test else target_dataset
    top_k = 1 if is_test else 3

    # ── KG / 결과 경로 분기 ───────────────────────────────────────────────────
    if use_ensemble:
        kg_store_path = KG_STORE_DIR / "kg_store_ontology.json"
        result_stem = "eval_results_ensemble_hard" if args.hard else "eval_results_ensemble"
    elif use_naive:
        kg_store_path = "N/A (Text files)"
        result_stem = "eval_results_naive_hard" if args.hard else "eval_results_naive"
    elif use_ontology:
        kg_store_path = KG_STORE_DIR / "kg_store_ontology.json"
        result_stem = "eval_results_ontology_hard" if args.hard else "eval_results_ontology"
    else:
        kg_store_path = KG_STORE_DIR / "kg_store.json"
        result_stem = "eval_results_hard" if args.hard else "eval_results"

    result_path_base = EVAL_RESULTS_DIR / result_stem
    result_path = result_path_base.with_stem(result_stem + "_test") if is_test else result_path_base
    result_path = result_path.with_suffix(".csv")

    def log(msg):
        print(msg, flush=True)

    def elapsed(t0):
        s = time.time() - t0
        return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"

    if use_ensemble:
        kg_label = "앙상블 RAG (Naive + 온톨로지 KG)"
    elif use_naive:
        kg_label = "Naive RAG"
    else:
        kg_label = "온톨로지 KG" if use_ontology else "일반 KG"
    mode_label = "[TEST MODE] " if is_test else ""
    log(f"=== {mode_label}Phase 4: Ragas 기반 KG-RAG 하이브리드 자동 평가 ({kg_label}) ===")
    log("")
    log("── 경로 상수 ──────────────────────────────────────────")
    log(f"  BASE_DIR      : {BASE_DIR}")
    log(f"  DATA_DIR      : {DATA_DIR}")
    log(f"  CONFIG_PATH   : {CONFIG_PATH}")
    log(f"  KG_STORE_PATH : {kg_store_path}")
    log(f"  RESULT_PATH   : {result_path}")
    log("────────────────────────────────────────────────────────")
    log("")
    if is_test:
        log(f"  -> 테스트 모드: {len(dataset)}문항, similarity_top_k={top_k}")

    total_t0 = time.time()

    t0 = time.time()
    log("[1/4] 모델 및 KG 로딩 중...")
    api_key = load_api_key()
    init_models(api_key)
    
    if use_ensemble:
        log("  -> [Ensemble RAG] VectorStoreIndex 및 PropertyGraphIndex(온톨로지) 로딩 중...")
        parsed_files = list(PARSED_DOCS_DIR.glob("*_parsed.txt"))
        docs = SimpleDirectoryReader(input_files=parsed_files).load_data()
        vector_index = VectorStoreIndex.from_documents(docs)
        kg_index = load_index(kg_store_path)
    elif use_naive:
        log("  -> [Naive RAG] 텍스트 문서 파싱 및 VectorStoreIndex 구축 중...")
        parsed_files = list(PARSED_DOCS_DIR.glob("*_parsed.txt"))
        docs = SimpleDirectoryReader(input_files=parsed_files).load_data()
        index = VectorStoreIndex.from_documents(docs)
    else:
        index = load_index(kg_store_path)
    log(f"[1/4] 완료 ({elapsed(t0)})")

    if use_ensemble:
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        kg_retriever = kg_index.as_retriever(
            include_text=True,
            retriever_mode="hybrid",
            similarity_top_k=top_k
        )
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, kg_retriever],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False
        )
        query_engine = RetrieverQueryEngine.from_args(
            fusion_retriever,
            text_qa_template=KG_QA_TEMPLATE,
        )
    elif use_naive:
        query_engine = index.as_query_engine(similarity_top_k=top_k)
    else:
        query_engine = index.as_query_engine(
            include_text=True,
            retriever_mode="hybrid",
            similarity_top_k=top_k,
            text_qa_template=KG_QA_TEMPLATE,
        )

    t0 = time.time()
    log(f"[2/4] 평가 데이터셋 추론 시작... ({len(dataset)}문항)")
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(dataset, 1):
        q = item["question"]
        q_t0 = time.time()
        log(f"\n  [{i}/{len(dataset)}] 질의: {q}")
        response = query_engine.query(q)
        ans = response.response
        questions.append(q)
        answers.append(ans)
        ground_truths.append(item["ground_truth"])
        ctx = [n.node.get_content() for n in response.source_nodes[:top_k]]
        contexts.append(ctx)
        log(f"    -> [답변 요약] {ans[:60]}..." if len(ans) > 60 else f"    -> [답변] {ans}")
        log(f"    -> [Context] {len(ctx)}개 (전체 검색: {len(response.source_nodes)}개) ({elapsed(q_t0)})")

    log(f"[2/4] 완료 ({elapsed(t0)})")

    hf_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    ragas_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, timeout=60)
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key, timeout=60)

    total_ctx = sum(len(c) for c in contexts)
    t0 = time.time()
    log(f"[3/4] Ragas Metric 점수 계산 중... (총 context {total_ctx}개, LLM 호출 비용 발생)")
    result = evaluate(
        hf_dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    log(f"[3/4] 완료 ({elapsed(t0)})")

    log("\n=== Ragas 평가 결과 (평균 점수) ===")
    log(str(result))

    t0 = time.time()
    df = result.to_pandas()
    json_path = result_path.with_suffix(".json")
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=4)

    log(f"\n[4/4] 결과 리포트 저장 완료 ({elapsed(t0)}):")
    log(f"  - CSV: {result_path}")
    log(f"  - JSON: {json_path}")
    log(f"\n총 소요 시간: {elapsed(total_t0)}")

if __name__ == "__main__":
    main()
