import streamlit as st
import os
import json
import itertools
import base64
import tempfile
import platform 
import time
from datetime import datetime
import pandas as pd
from io import BytesIO
import gc


# [필수 라이브러리]
from pdf2image import convert_from_path
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ==========================================================
# [0] 기본 설정 및 상수 정의
# ==========================================================
st.set_page_config(page_title="건설 CM AI 통합 솔루션 (Deep RAG)", page_icon="🏗️", layout="wide")

# 1. API 키 가져오기
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 치명적 오류: Google API Key가 없습니다.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Poppler 경로 (Windows 환경 대응)
system_name = platform.system()
if system_name == "Windows":
    # 로컬(내 컴퓨터)에서 돌릴 때만 경로 지정 (본인 경로로 수정 필요)
    # 잘 모르겠으면 일단 None으로 두세요. (환경변수에 있다면 작동함)
    POPPLER_PATH = None 
else:
    # Streamlit Cloud 등 서버(Linux)에서는 보통 경로 지정 불필요 (패키지로 설치됨)
    POPPLER_PATH = None

# 3. 데이터 경로
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"
RAW_DATA = []
# 5. 히스토리 파일 경로 설정
HISTORY_FILE = "chat_history.json"

def save_chat_history():
    """대화 내역 저장 시 엑셀 등 무거운 데이터는 제외하고 텍스트만 저장"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        serializable_msgs = []
        for m in st.session_state.messages:
            # 메시지가 딕셔너리 형태일 때 (AI 답변 등)
            if isinstance(m, dict):
                # 엑셀 데이터(excel_data)는 빼고 role과 content만 추출
                clean_msg = {
                    "role": m.get("role", "assistant"),
                    "content": m.get("content", "")
                }
            else:
                # 메시지가 객체 형태일 때 (사용자 질문 등)
                role = "user" if "Human" in str(type(m)) else "assistant"
                clean_msg = {
                    "role": role, 
                    "content": getattr(m, 'content', str(m))
                }
            
            serializable_msgs.append(clean_msg)
            
        # 텍스트만 모인 리스트를 파일로 저장
        json.dump(serializable_msgs, f, ensure_ascii=False, indent=4)

def load_chat_history():
    """저장된 파일이 있으면 불러오기"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

# --- 세션 상태 초기화 수정 ---
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history() # 빈 리스트 대신 파일 로드
# 4. 모델 설정
MODEL_NAME = "models/gemini-2.5-pro" 

# ==========================================================
# [1] 시스템 로딩 (검색 엔진 & 모델)
# ==========================================================
class SimpleHybridRetriever:
    """BM25(키워드) + Chroma(벡터) 결합 검색기"""
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        # 1. 병렬 검색 수행
        docs_bm25 = self.bm25.invoke(query)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        # 2. Chroma 결과 복원 (인덱스 -> 원본 텍스트)
        real_docs_chroma = []
        for doc in (docs_c1 + docs_c2):
            try:
                idx = int(doc.page_content) 
                original_item = self.raw_data[idx] 
                content = original_item.get('content', '').strip()
                source = original_item.get('source', '').strip()
                article = original_item.get('article', '').strip()
                full_text = f"[{source}] {content}"
                new_doc = Document(page_content=full_text, metadata={"source": source, "article": article})
                real_docs_chroma.append(new_doc)
            except:
                continue

        # 3. 결과 통합 및 중복 제거
        combined = []
        seen_ids = set()
        for d in itertools.chain(docs_bm25, real_docs_chroma):
            key = d.page_content[:30] # 내용 앞부분으로 중복 체크
            if key not in seen_ids:
                combined.append(d)
                seen_ids.add(key)
        return combined[:200] # 1차적으로 넉넉하게 반환

@st.cache_resource
def load_search_system():
    global RAW_DATA
    if not os.path.exists(JSON_DATA_PATH):
        st.error("❌ JSON 데이터 파일이 없습니다.")
        st.stop()
        
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        RAW_DATA = json.load(f)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    if not os.path.exists(DB_PATH_1) or not os.path.exists(DB_PATH_2):
        st.error("❌ DB 폴더가 없습니다.")
        st.stop()

    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    retriever1 = store1.as_retriever(search_kwargs={"k": 100})
    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    retriever2 = store2.as_retriever(search_kwargs={"k": 100})

    docs = []
    for item in RAW_DATA:
        content = item.get('content', '').strip()
        source = item.get('source', '').strip()
        if not content: continue
        doc = Document(page_content=f"[{source}] {content}", metadata={"source": source, "article": item.get('article', '')})
        docs.append(doc)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 150
    hybrid_retriever = SimpleHybridRetriever(bm25_retriever, retriever1, retriever2, RAW_DATA)
    
    # --- [수정 구간] Cross-Encoder (Reranker) 메모리 최적화 로드 ---
    try:
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L-2-v2", 
            device="cpu", # 무료 서버는 CPU 강제 사용
            model_kwargs={"low_cpu_mem_usage": True} # 메모리 점유 최소화
        )
    except Exception as e:
        st.warning(f"⚠️ Reranker(TinyBERT) 로드 실패: {e}. 기본 검색 모드로 동작합니다.")
        reranker = None 

    return hybrid_retriever, reranker

with st.spinner("🚀 AI 5단계 심층 검색 엔진 시동 중..."):
    try:
        hybrid_retriever, reranker_model = load_search_system()
    except Exception as e:
        st.error(f"시스템 로딩 실패: {e}")
        st.stop()

# LLM 초기화
safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
llm_text = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)
llm_vision = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)

# ==========================================================
# [2] Deep RAG 파이프라인 (5단계 로직 구현)
# ==========================================================

# (1) 쿼리 확장 (Query Expansion)
expansion_prompt = ChatPromptTemplate.from_template("""
당신은 건설/건축 검색 최적화 AI입니다.
사용자 질문을 분석하여 검색 정확도를 높일 수 있는 **'확장 검색어'** 3개를 생성하세요.
건설 표준 시방서, 법규 용어, 동의어를 포함해야 합니다.

[사용자 질문]: {question}

[출력 형식]: 질문 | 키워드1, 키워드2, 키워드3
(설명 없이 위 형식으로만 출력하세요)
""")
expansion_chain = expansion_prompt | llm_text | StrOutputParser()

def get_expanded_queries(original_query):
    """(1단계) 사용자 질문을 확장하여 리스트로 반환"""
    try:
        expanded_str = expansion_chain.invoke({"question": original_query})
        if "|" in expanded_str:
            base, keywords = expanded_str.split("|", 1)
            queries = [base.strip()] + [k.strip() for k in keywords.split(",")]
        else:
            queries = [original_query]
        return queries[:4] # 최대 4개까지만 사용 (속도 조절)
    except:
        return [original_query]

# (2)~(4) 하이브리드 검색 + 재순위화 + Top-K 필터링
def retrieve_and_rerank(query, top_k=50):
    # Step 1: 쿼리 확장
    expanded_queries = get_expanded_queries(query)
    
    # Step 2: 하이브리드 검색 (확장된 쿼리 각각 수행)
    all_docs = []
    seen_contents = set()
    
    for q in expanded_queries:
        docs = hybrid_retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
    
    if not all_docs: return []

    # --- [수정 구간] Reranker가 없는 경우를 대비한 안전 장치 ---
    # reranker_model이 None이면 정밀 재순위화 단계를 건너뛰고 바로 결과를 반환합니다.
    if reranker_model is None:
        return all_docs[:top_k]

    # Step 3: 정밀 재순위화 (Cross-Encoder)
    pairs = [[query, doc.page_content] for doc in all_docs]
    scores = []
    batch_size = 16 # 메모리 확보를 위해 배치 사이즈를 줄임 (기존 32)
    
    try:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_scores = reranker_model.predict(batch)
            scores.extend(batch_scores)
        
        scored_docs = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
        # Step 4: Top-K 필터링
        final_top_k = [doc for doc, score in scored_docs[:top_k]]
    except Exception as e:
        st.error(f"Reranking 과정 중 오류 발생: {e}")
        return all_docs[:top_k]
        
    return final_top_k

# (5) 답변 생성 (유연한 프롬프트)
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm_text).pipe(StrOutputParser())

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 베테랑 건설 사업 관리자(CM)이자 시공 기술사입니다.
    사용자의 질문에 대해 아래 [Context](검색된 법규/시방서)를 참고하여 답변해야 합니다.

    [답변 규칙]
    1. **우선 순위:** [Context]에 구체적인 절차나 기준이 있다면 반드시 그것을 근거로 답변하세요.
    2. **일반 지식 활용:** 만약 [Context]에 '해결 방안'이나 '구체적 공법'이 부족하다면, 
       **"제공된 법규 데이터에는 구체적 방법이 명시되지 않았으나, 일반적인 시공 기준에 따르면..."** 이라고 언급한 뒤, 당신이 알고 있는 **표준 시방서 및 공학적 지식**을 동원해 해결책을 제시하세요.
    3. 절대 "모른다"고 끝내지 말고, 실무적인 조언을 제공하세요.
    4. 출처가 있다면 [출처: ...] 형태로 명시하세요.

    [Context]
    {context}
    """),
    ("human", "질문: {question}")
])

def format_docs(docs):
    return "\n\n".join([f"<출처: {d.metadata.get('source')} / {d.metadata.get('article')}>\n{d.page_content}" for d in docs])

# 최종 RAG 체인 (Top-50 적용)
rag_chain = (
    {"context": RunnableLambda(lambda x: retrieve_and_rerank(x, top_k=50)) | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm_text | StrOutputParser()
)

# ==========================================================
# [3] Vision AI (도면 분석용)
# ==========================================================
def analyze_page_detail(image_base64, query, retrieved_docs):
    # 검색된 법규 데이터를 텍스트로 변환
    laws_context = "\n".join([f"[{d.metadata.get('source')}] {d.page_content}" for d in retrieved_docs])
    
    if not laws_context.strip():
        laws_context = "관련된 구체적 법규 데이터가 없습니다. 일반 기술 지식을 바탕으로 분석하세요."

    # [수정 방향] 사용자께서 주신 지침을 변수로 고정하여 프롬프트에 강제 주입
    structural_guideline = """
    [최우선 관리 항목: 구조 안전성 및 보강 공법 검토 지침]
    1. 구조 안전성 및 보강 공법 검토 (최우선 관리 항목)
       - 본 프로젝트는 노후 건물(1974년 준공) 상부에 1개 층을 수직 증축하고 3개 동을 연결하는 고난도 공사임.
    2. 기존 구조물 상태 평가 및 보강 선후행 관리:
       - 철거 공사 착수 전, 기존 슬래브, 보, 기둥의 균열, 처짐, 내력 저하 여부 점검 및 준공 도면 일치 여부 보고 확인.
       - 증축 공사(철골 세우기) 전, 하부 구조물의 연직 보강(단면 증타, 철판 보강) 및 내진 보강(탄소섬유 보강) 완료 여부 확인.
    3. 주요 보강 공법에 대한 적정성 검토:
       - 기둥/보 보강: 기둥 단면 증타 및 철판 압착 보강(SV-401) 시공 시, 기존 콘크리트 면의 치핑(Chipping) 상태와 신구 접착제 성능, 무수축 몰탈의 충전 밀실도 검증.
       - 슬래브 탄소섬유 보강: 탄소섬유 시트(SK-N600 등) 부착 시, 바탕면 처리 상태(평활도)와 프라이머 도포 적정성, 부착 강도 시험(Pull-off test) 계획 수립 여부.
    4. 신구 접합부 및 이질 재료 연결 상세:
       - 기존 RC 기둥 상부 신설 철골 기둥(SC1, SC2 등) 정착용 베이스 플레이트 및 케미컬 앵커(Hilti RE500 등) 인발 내력 시험 성적서 확인 및 간섭 검토.
       - 철골보(H-Beam)와 기존 콘크리트 보 연결 부위 전단 접합 상세(Shear Connection) 시공성 검토.
    """

    prompt_text = f"""
        당신은 베테랑 건설 사업 관리자(CM)이자 시공 기술사입니다. 
        당신의 최우선 임무는 제공된 **[구조 안전 지침]**과 도면을 대조하여 팩트 중심의 검토 보고서를 작성하는 것입니다.
        
        [질문/검토 요청]: {query}
        
        {structural_guideline}
        
        [참고 법규/기준(DB)]:
        {laws_context}
        
        [분석 지침]:
        1. **지침 기반 필터링:** 도면 분석 시 반드시 위 1~4번 지침을 기준으로 위반 사항이나 누락 사항을 찾으십시오.
        2. **수리적 팩트 체크:** 앵커의 정착 깊이, 보강 두께, EL/FL 값 등을 도면에서 찾아 계산식을 명시하십시오.
        3. **상세도 검토:** SC1, SC2 등 신설 기둥과 기존 RC 기둥의 접합부(Base Plate) 상세가 지침(Hilti RE500 등)과 일치하는지 확인하십시오.

        [답변 구조]:
        - **중대 위반/위험 항목 (구조 보강 특화):**
            * **[근거]:** (위 지침 중 해당 항목 번호 명시)
            * 현황: (도면 위치 및 확인된 수치/표기)
            * 문제점: (지침 대비 미흡한 점이나 예상되는 구조적 결함)
            * 개선안: (구체적 보강 공법이나 시공 시 보완 대책)
        - **기술적 제언:** 구조 외에 시공 전문가로서 제언하는 사항
        """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ])
    
    try:
        response = llm_vision.invoke([message])
        return response.content
    except Exception as e:
        return f"분석 오류 발생: {e}"

def generate_final_report(file_name, page_results):
    raw_data = ""
    for item in page_results:
        raw_data += f"\n[Page {item['page']}]: {item['content']}\n"
    
    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    
    # 보고서 생성 시에도 1974년 노후 건물 증축이라는 맥락을 유지하게 함
    prompt = f"""
    당신은 건설사업관리단장(CM단장)입니다. 
    1974년 준공 노후 건물의 수직 증축 및 보강 공사라는 특수성을 고려하여 '최종 시공 품질/안전 검토 보고서'를 작성하세요.

    [작성 가이드]
    - '구조 안전성 확보'와 '보강 공법의 실무적 적정성'을 가장 강조하십시오.
    - 프로젝트명은 별도 언급 없으면 '{file_name}'으로 기재하세요.

    [분석 데이터]
    {raw_data}
    
    [보고서 형식]
    1. 도면명: {file_name}
    2. 작성 일자: {current_date}
    3. 작성자: AI 건설 지원 시스템 (구조 안전 특화)
    4. 검토 총평: (노후 건축물 증축에 따른 구조적 리스크와 보강 대책 요약)
    5. 주요 검토 내용 (항목별 요약): ...
    """
    return llm_text.invoke(prompt).content


    """
    AI가 작성한 보고서 텍스트를 파싱하여 맥킨지 스타일의 엑셀 데이터로 변환
    """
    # AI에게 엑셀용 표 데이터를 따로 추출하도록 요청
    excel_prompt = f"""


    [출력 규칙]
    - 반드시 '도면정보 | 현황 | 문제점 | 개선안 | 체크여부' 형식으로 출력하세요.
    - 구분자 '|'를 사용하고 다른 설명은 일절 생략하세요.

    [보고서 내용]
    {report_content}
    """
    
    try:
        raw_data = llm_text.invoke(excel_prompt).content
        rows = []
        for line in raw_data.split('\n'):
            if '|' in line and '도면정보' not in line: # 헤더 제외 데이터만
                rows.append([item.strip() for item in line.split('|')])
        
        # 데이터프레임 생성
        df = pd.DataFrame(rows, columns=['도면정보', '현황', '문제점', '개선안', '체크여부'])
        
        # 엑셀 파일 생성 (메모리 버퍼)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='도면검토체크리스트')
            
            workbook  = writer.book
            worksheet = writer.sheets['도면검토체크리스트']

            # 맥킨지 스타일 서식 (남색 헤더, 흰색 글자, 맑은 고딕)
            header_format = workbook.add_format({
                'bold': True, 'font_name': '맑은 고딕', 'font_size': 11,
                'bg_color': '#003366', 'font_color': 'white',
                'border': 1, 'align': 'center', 'valign': 'vcenter'
            })
            body_format = workbook.add_format({
                'font_name': '맑은 고딕', 'font_size': 10,
                'border': 1, 'valign': 'vcenter', 'text_wrap': True
            })

            # 서식 적용 및 열 너비 조정
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 30, body_format)
            
            worksheet.set_row(0, 25) # 헤더 높이

        return output.getvalue()
    except:
        return None
    
def create_excel_report(report_content):
    """
    최종 수정: B2 시작, 일정한 행 높이, 도면정보 줄바꿈 및 요약 적용
    """
    excel_prompt = f"""
    당신은 건설 데이터 전문가입니다. 아래 [보고서 내용]을 바탕으로 현장 기술자용 체크리스트를 만드세요.

    [작성 규칙]
    1. 도면 정보(페이지): '파일명.pdf'와 '(페이지/부위)' 사이에 반드시 'NL'이라는 글자를 넣어 구분하세요.
    2. 현황 및 문제점 / 개선안(보완대책): 핵심 키워드 중심으로 아주 간결하게 '요약'해서 작성하세요.
    
    [표 구성 및 출력 형식]
    형식: 도면 정보(페이지) | 현황 및 문제점 | 개선안(보완대책) | 체크
    구분자 '|'를 사용하고 데이터만 출력하세요.

    [보고서 내용]
    {report_content}
    """
    
    try:
        raw_data = llm_text.invoke(excel_prompt).content
        rows = []
        for line in raw_data.split('\n'):
            if '|' in line and '도면 정보' not in line:
                parts = [item.strip() for item in line.split('|')]
                if len(parts) >= 3:
                    drawing_info = parts[0].replace("NL", "\n")
                    rows.append([drawing_info, parts[1], parts[2], "□ 미확인"])
        
        df = pd.DataFrame(rows, columns=['도면 정보(페이지)', '현황 및 문제점', '개선안(보완대책)', '체크'])
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # [수정] B2 셀부터 시작하도록 startrow=1, startcol=1 설정
            df.to_excel(writer, index=False, sheet_name='검토체크리스트', startrow=1, startcol=1)
            
            workbook  = writer.book
            worksheet = writer.sheets['검토체크리스트']

            # 서식 정의
            header_fmt = workbook.add_format({
                'bold': True, 'font_name': '맑은 고딕', 'font_size': 11,
                'bg_color': '#003366', 'font_color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter'
            })
            
            center_bold_fmt = workbook.add_format({
                'bold': True, 'font_name': '맑은 고딕', 'font_size': 10, 'border': 1,
                'align': 'center', 'valign': 'vcenter', 'text_wrap': True
            })
            center_bold_gray_fmt = workbook.add_format({
                'bold': True, 'font_name': '맑은 고딕', 'font_size': 10, 'border': 1,
                'align': 'center', 'valign': 'vcenter', 'text_wrap': True, 'bg_color': '#F2F5F9'
            })
            
            left_vcenter_fmt = workbook.add_format({
                'font_name': '맑은 고딕', 'font_size': 10, 'border': 1,
                'align': 'left', 'valign': 'vcenter', 'text_wrap': True
            })
            left_vcenter_gray_fmt = workbook.add_format({
                'font_name': '맑은 고딕', 'font_size': 10, 'border': 1,
                'align': 'left', 'valign': 'vcenter', 'text_wrap': True, 'bg_color': '#F2F5F9'
            })

            # 열 너비 설정 (B열부터 시작하므로 A열은 비워둠)
            worksheet.set_column('A:A', 3)   # 왼쪽 여백
            worksheet.set_column('B:B', 22)  # 도면 정보
            worksheet.set_column('C:C', 45)  # 현황 및 문제점
            worksheet.set_column('D:D', 50)  # 개선안
            worksheet.set_column('E:E', 12)  # 체크

            # [수정] 데이터 쓰기 및 행 높이 통일
            # 고정 행 높이 설정 (가독성을 위해 45~50 정도가 적당합니다)
            row_height = 45 
            
            for row_num, data in enumerate(rows):
                is_even = row_num % 2 == 1
                r_idx = row_num + 2 # B2가 헤더이므로 데이터는 3행(index 2)부터 시작
                
                # 행 높이 일정하게 고정
                worksheet.set_row(r_idx, row_height)
                
                fmt_center = center_bold_gray_fmt if is_even else center_bold_fmt
                fmt_left = left_vcenter_gray_fmt if is_even else left_vcenter_fmt
                
                # 열 위치를 하나씩 밀어서(1, 2, 3, 4) 작성
                worksheet.write(r_idx, 1, data[0], fmt_center) # B열
                worksheet.write(r_idx, 2, data[1], fmt_left)   # C열
                worksheet.write(r_idx, 3, data[2], fmt_left)   # D열
                worksheet.write(r_idx, 4, data[3], fmt_center) # E열

            # 헤더 다시 쓰기 (B2 셀부터)
            worksheet.set_row(1, 32) # 헤더 행 높이
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(1, col_num + 1, value, header_fmt)
            
            # 드롭박스 위치 조정 (E열)
            worksheet.data_validation(2, 4, len(df)+1, 4, {'validate': 'list', 'source': ['□ 미확인', '✅ 확인완료']})
            worksheet.freeze_panes(2, 0) # 2행까지 고정 (B2 헤더 보이게)

        return output.getvalue()
    except Exception as e:
        st.error(f"Excel 생성 오류: {e}")
        return None
    
# ==========================================================
# [4] 웹 UI (Streamlit)
# ==========================================================
st.title("🏗️ 건설 CM 전문 AI (Deep RAG + Vision)")

# 세션 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_image_base64" not in st.session_state:
    st.session_state.current_image_base64 = None

# --- [사이드바] 수정 부분 ---
with st.sidebar:
    st.header("📂 도면 투입구")
    uploaded_files = st.file_uploader("PDF 도면 업로드", type=["pdf"], accept_multiple_files=True)
    
    st.markdown("---")
    
    # 3가지 명확한 모드 정의
    mode_options = ["⚖️ 법규 DB 검색", "💬 순수 Gemini 지식"]
    if uploaded_files or st.session_state.current_image_base64:
        mode_options.insert(0, "📂 도면 기반 질문")

    st.subheader("🤖 질문 모드")
    search_mode = st.radio("모드 선택", mode_options, index=0)

    # 대화 삭제 버튼 (파일까지 삭제)
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()


        
# --- [메인] 도면 처리 로직 (1번 방법 적용 수정본) ---
if uploaded_files:
    for target_file in uploaded_files:
        if target_file.name not in st.session_state.processed_files:
            with st.status(f"📄 '{target_file.name}' 분석 중...", expanded=True) as status:
                # 1. 임시 파일 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(target_file.read())
                    tmp_path = tmp_file.name 
                
                try:
                    # [수정된 부분] pdf_info_to_dict 대신 pdfinfo_from_path 사용
                    from pdf2image import pdfinfo_from_path
                    
                    # PDF 정보 가져오기
                    info = pdfinfo_from_path(tmp_path, poppler_path=POPPLER_PATH)
                    total_pages = info["Pages"] # 페이지 수 추출
                    
                    page_results = []
                    progress = st.progress(0)

                    # 2. Vision 분석 루프
                    for i in range(total_pages):
                        curr_page = i + 1
                        progress.progress(curr_page / total_pages, text=f"🔍 {curr_page}/{total_pages} 페이지 정밀 진단 중...")
                        
                        # [핵심] 해당 페이지만 메모리에 로드 + 해상도 조절(size)로 메모리 절약
                        # size=(1200, None)은 가로를 1200px로 맞추고 세로는 비율에 맞게 조정합니다.
                        page_images = convert_from_path(
                            tmp_path, 
                            first_page=curr_page, 
                            last_page=curr_page,
                            size=(1200, None), 
                            poppler_path=POPPLER_PATH
                        )
                        
                        if not page_images:
                            continue
                         
                        page_img = page_images[0]
                        
                        # 이미지 base64 변환
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                            # quality를 75~80으로 낮추면 메모리 점유율이 더 내려갑니다.
                            page_img.save(tmp_img.name, "JPEG", quality=80)
                            with open(tmp_img.name, "rb") as f:
                                img_base64 = base64.b64encode(f.read()).decode("utf-8")
                        
                        st.session_state.current_image_base64 = img_base64 # 최신 이미지 유지
                        
                        # 분석 실행
                        res = analyze_page_detail(img_base64, "위험 요소 식별", [])
                        page_results.append({"page": curr_page, "content": res})
                        
                        # [핵심] 메모리 강제 해제
                        # 변수 참조를 제거하고 가비지 컬렉터를 호출합니다.
                        del page_img
                        del page_images
                        gc.collect() 

                except Exception as e:
                    st.error(f"변환/분석 중 오류 발생: {e}")
                    continue
                finally:
                    # 분석 완료 후 임시 PDF 파일 삭제 시도
                    if os.path.exists(tmp_path):
                        try: os.remove(tmp_path)
                        except: pass
                
                # 3. 종합 보고서 작성 및 파일별 엑셀 생성
                status.write("📝 종합 보고서 작성 중...")
                report = generate_final_report(target_file.name, page_results)
                current_excel_data = create_excel_report(report)

                st.session_state.processed_files.add(target_file.name)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": report,
                    "excel_data": current_excel_data,
                    "file_name": target_file.name
                })

                progress.empty()
                status.update(label="✅ 분석 완료", state="complete")


# --- [채팅창 영역 하단 수정] ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # [수정] 메시지 안에 엑셀 데이터가 들어있는 경우에만 버튼 생성
        if msg.get("excel_data") is not None:
            st.download_button(
                label=f"📥 {msg['file_name']} 검토결과(엑셀) 다운로드",
                data=msg["excel_data"],
                file_name=f"검토결과_{msg['file_name']}_{datetime.now().strftime('%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"btn_dl_{idx}" # 고유 ID 부여로 오류 해결
            )

# --- [메인 채팅 처리 로직] ---
if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 질문 기록
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # [Option 1] 도면 기반 질문 (도면 이미지 + DB 검색)
        if search_mode == "📂 도면 기반 질문":
            with st.status("🔍 도면 및 법규 교차 분석 중..."):
                corrected_query = spacing_chain.invoke({"question": prompt})
                relevant_docs = retrieve_and_rerank(corrected_query, top_k=10) 
                response = analyze_page_detail(st.session_state.current_image_base64, prompt, relevant_docs)
        
        # [Option 2] 법규 DB 검색 (RAG 전용)
        elif search_mode == "⚖️ 법규 DB 검색":
            with st.status("🧠 DB 내 법규/시방서 검색 중..."):
                response = rag_chain.invoke(prompt)

        # [Option 3] 순수 Gemini 지식 (문맥 유지 - 이 부분이 포인트!)
        else:
            with st.spinner("Gemini가 대화 내역을 읽고 답변 중..."):
                # ★ prompt만 보내는 게 아니라 전체 messages를 보냅니다.
                # 이를 통해 앞서 생성된 '도면 분석 보고서' 내용을 제미나이가 인지합니다.
                res_object = llm_text.invoke(st.session_state.messages)
                response = res_object.content

        # 2. 결과 출력 및 저장
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_history() # 파일에 즉시 영구 저장