"""
PDF → 텍스트 추출 스크립트 (pdfplumber 기반)

동작 흐름:
  1. PDF 파일을 페이지 단위로 열어서 텍스트를 추출한다.
  2. 페이지 안에 표(table)가 있으면 별도로 추출해 Markdown 테이블 형식으로 변환한다.
  3. 텍스트 + Markdown 표를 합쳐서 _parsed.txt 파일로 저장한다.

왜 pdfplumber를 쓰나?
  - PyMuPDF 등 다른 라이브러리보다 표 영역 감지·추출 품질이 높다.
  - extract_tables()가 셀 단위 리스트를 반환해 직접 가공하기 쉽다.
  - 읽어야 하는 도메인 지식(주금공 pdf)에 표가 많아서 이 방법을 사용함.

출력 형식 예시:
  일반 본문 텍스트...

  [TABLE]
  | 항목 | 내용 |
  | --- | --- |
  | 금리 | 연 3.5% |
  [/TABLE]
"""

import pdfplumber          # PDF 파싱 핵심 라이브러리
import re                  # 현재는 직접 사용하지 않지만 추후 텍스트 정제용으로 import 유지
from pathlib import Path   # 운영체제에 무관하게 파일 경로를 다루기 위해 사용

# ─────────────────────────────────────────────
# 경로 설정
#   이 파일의 위치: src/parse/pdf_to_text.py
#   __file__           → src/parse/pdf_to_text.py (절대 경로)
#   .parent            → src/parse/
#   .parent.parent     → src/
#   .parent.parent.parent → 프로젝트 루트 (KG_LLM/)
#   / "data"           → KG_LLM/data/
# ─────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent.parent / "data"
RAW_DOCS_DIR    = DATA_DIR / "raw_docs"
PARSED_DOCS_DIR = DATA_DIR / "parsed_docs"


def table_to_markdown(table: list[list]) -> str:
    """
    pdfplumber가 반환한 2차원 리스트(표)를 Markdown 테이블 문자열로 변환한다.

    매개변수:
        table: [[셀, 셀, ...], [셀, 셀, ...], ...] 형태의 2차원 리스트.
               첫 번째 행이 헤더로 사용된다.

    반환값:
        Markdown 테이블 문자열. 예:
            | 항목 | 내용 |
            | --- | --- |
            | 금리 | 연 3.5% |
        표가 비어 있으면 빈 문자열("")을 반환한다.
    """

    def clean(cell):
        """
        셀 값 정규화 함수.
        - None(값이 없는 셀)은 빈 문자열로 치환
        - 셀 안의 줄바꿈(\n)은 공백으로 변환 (Markdown 테이블에서 줄바꿈은 표를 깨뜨리기 때문)
        - 앞뒤 공백 제거
        """
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    # 모든 셀에 clean() 적용 → 표 전체를 깨끗한 문자열 2차원 리스트로 만든다
    rows = [[clean(c) for c in row] for row in table]

    # 완전히 빈 행(모든 셀이 빈 문자열인 행) 제거
    # PDF에서 병합 셀이나 레이아웃 오류로 인해 빈 행이 생기는 경우가 있다
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""  # 유효한 행이 하나도 없으면 빈 문자열 반환

    # 각 행의 열 수가 다를 수 있으므로, 가장 긴 행 기준으로 열 수를 통일
    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]  # 짧은 행은 빈 셀로 패딩

    # Markdown 테이블 구성
    header = rows[0]                   # 첫 번째 행 → 헤더
    separator = ["---"] * col_count    # 헤더와 본문 사이 구분선 (Markdown 문법)
    body = rows[1:]                    # 나머지 행 → 데이터 본문

    lines = []
    lines.append("| " + " | ".join(header) + " |")      # 헤더 행
    lines.append("| " + " | ".join(separator) + " |")   # 구분선
    for row in body:
        lines.append("| " + " | ".join(row) + " |")     # 데이터 행

    return "\n".join(lines)


def merge_spanning_cells(table: list[list]) -> list[list]:
    """
    세로 방향으로 병합된 셀(rowspan)을 처리한다.

    문제 배경:
        PDF에서 표를 추출할 때 pdfplumber는 세로 병합 셀(예: 위아래 2칸이
        하나로 합쳐진 셀)을 첫 번째 행에만 값을 넣고 나머지 행에는 None을 넣는다.
        예)
          원본 표:        pdfplumber 추출 결과:
          ┌───┬───┐        [["구분", "내용"],
          │구분│내용│        ["금리", "3.5%"],
          │   ├───┤   →    [None, "4.0%"],   ← "구분" 셀이 None
          │   │내용│        [None, "4.5%"]]  ← "구분" 셀이 None
          └───┴───┘

    해결 방법:
        레이블 역할을 하는 왼쪽 열(첫 1~2개 열)의 None 값은
        바로 위 행의 값으로 채운다(forward-fill).
        단, 값을 담는 오른쪽 열은 건드리지 않는다(집계값이 0이거나 의도적으로 비어 있을 수 있음).
        또한 헤더 행(row 0)은 채움 대상에서 제외한다.

    매개변수:
        table: pdfplumber의 extract_tables()가 반환한 원본 2차원 리스트

    반환값:
        병합 셀이 채워진 2차원 리스트
    """
    if not table:
        return table  # 빈 표이면 그대로 반환

    # 각 행의 열 수가 다를 수 있으므로 가장 긴 행 기준으로 None으로 패딩
    col_count = max(len(r) for r in table)
    padded = [r + [None] * (col_count - len(r)) for r in table]

    # 레이블 컬럼 범위 결정:
    #   열이 3개 이상인 표 → 첫 2열이 레이블(대분류, 소분류)
    #   열이 2개인 표     → 첫 1열만 레이블
    label_cols = set(range(min(2, col_count - 1))) if col_count >= 3 else {0}

    # 헤더 행은 그대로 결과에 넣고, 이후 행부터 채움 처리
    result = [list(padded[0])]  # padded[0] = 헤더 행 복사

    for row in padded[1:]:      # 헤더 이후 데이터 행을 순서대로 처리
        new_row = []
        for ci, cell in enumerate(row):
            # 레이블 컬럼이고 값이 None이면 → 바로 위 행(result[-1])의 같은 열 값으로 채움
            if cell is None and ci in label_cols:
                new_row.append(result[-1][ci])
            else:
                new_row.append(cell)  # 그 외에는 원래 값 유지(None 포함)
        result.append(new_row)

    return result


def extract_pdf(pdf_path: Path) -> str:
    """
    PDF 파일 하나를 열어 전체 내용을 텍스트로 추출한다.

    처리 순서 (페이지 단위):
        1. page.extract_text()  → 일반 텍스트(단락, 제목 등) 추출
        2. page.extract_tables() → 표 영역을 2차원 리스트로 추출
        3. 표는 merge_spanning_cells → table_to_markdown 순으로 변환 후
           [TABLE]...[/TABLE] 마커로 감싸서 본문 뒤에 추가

    왜 텍스트와 표를 따로 추출하나?
        pdfplumber의 extract_text()는 표 안의 텍스트도 포함하지만
        셀 구분 없이 뭉쳐서 나온다. 표를 별도로 extract_tables()로 뽑으면
        셀 구조를 유지한 채로 Markdown으로 깔끔하게 변환할 수 있다.
        (텍스트 추출에서 표 내용이 중복되더라도 LLM 파이프라인에서 무해하다.)

    매개변수:
        pdf_path: 읽을 PDF 파일의 Path 객체

    반환값:
        모든 페이지의 텍스트 + Markdown 표를 합친 하나의 문자열
    """
    output_parts = []  # 페이지별로 추출한 내용을 순서대로 담을 리스트

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            # ── 1. 일반 텍스트 추출 ──────────────────────────────────────
            # extract_text()가 None을 반환할 수 있으므로 or ""로 방어
            text = page.extract_text() or ""
            if text.strip():                          # 공백만 있는 페이지는 무시
                output_parts.append(text.strip())

            # ── 2. 표 추출 및 Markdown 변환 ──────────────────────────────
            tables = page.extract_tables()            # 페이지 안의 모든 표를 리스트로 반환
            for table in tables:
                filled = merge_spanning_cells(table)  # 세로 병합 셀 채우기
                md = table_to_markdown(filled)        # Markdown 문자열로 변환
                if md:
                    # [TABLE]...[/TABLE] 마커: 후처리 파이프라인에서
                    # 표 영역만 선택적으로 파싱하거나 제거할 수 있도록 표시
                    output_parts.append("\n[TABLE]\n" + md + "\n[/TABLE]\n")

    # 각 파트를 빈 줄 두 개(\n\n)로 구분해 하나의 문자열로 합침
    return "\n\n".join(output_parts)


def main():
    """
    처리할 PDF 파일 목록을 순서대로 읽어 파싱하고 결과를 .txt 파일로 저장한다.

    입력:  data/<파일명>.pdf
    출력:  data/<파일명>_parsed.txt

    파일이 존재하지 않으면 해당 파일은 건너뛰고(SKIP) 다음으로 진행한다.
    """
    # 처리할 PDF 파일 이름 목록 (data/ 디렉토리 기준 상대 이름)
    # RAW_DOCS_DIR 내의 모든 PDF 파일을 검색합니다.
    pdf_paths = list(RAW_DOCS_DIR.glob("*.pdf"))

    if not pdf_paths:
        print("[SKIP] 처리할 PDF 파일이 없습니다.")
        return

    for pdf_path in pdf_paths:
        filename = pdf_path.name


        # 파일 존재 여부 확인 — 없으면 경고 메시지만 출력하고 다음 파일로 넘어감
        if not pdf_path.exists():
            print(f"[SKIP] 파일 없음: {pdf_path}")
            continue

        print(f"[처리 중] {filename} ...")
        text = extract_pdf(pdf_path)

        # 출력 파일 경로: 원본 파일 이름에 _parsed.txt 접미사 추가
        # 예) 보금자리론 업무처리기준.pdf → 보금자리론 업무처리기준_parsed.txt
        out_path = PARSED_DOCS_DIR / (pdf_path.stem + "_parsed.txt")
        out_path.write_text(text, encoding="utf-8")   # UTF-8로 저장 (한글 깨짐 방지)
        print(f"[완료] {out_path.name}  ({len(text):,} chars)")


if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 때만 main()이 호출된다.
    # 다른 모듈에서 import해도 자동으로 실행되지 않도록 보호하는 관용 패턴이다.
    main()
