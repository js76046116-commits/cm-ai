import os
import json
import time
from tqdm import tqdm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ==========================================================
# [설정] 경로
# ==========================================================
INPUT_JSON_PATH = r"C:\Users\owner\myvenv\legal_data_total_vlm.json"
# 저장할 폴더 이름 2개
DB_PATH_1 = r"C:\Users\owner\myvenv\chroma_db_part1"
DB_PATH_2 = r"C:\Users\owner\myvenv\chroma_db_part2"

os.environ["GOOGLE_API_KEY"] = "AIzaSyCYDsHspn7XQm5pcGi6iKZVThqiNp_Xm4M"

def create_split_vector_db():
    print("🚀 [분할 모드] 벡터 DB를 2개로 쪼개서 만듭니다...")

    # 1. JSON 로드
    if not os.path.exists(INPUT_JSON_PATH):
        print("❌ JSON 파일이 없습니다.")
        return
    
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_len = len(data)
    print(f"📊 총 데이터: {total_len}개")
    
    # 2. 데이터를 정확히 반으로 나누기
    mid_index = total_len // 2
    data_part1 = data[:mid_index]
    data_part2 = data[mid_index:]
    
    print(f"   - Part 1: {len(data_part1)}개 -> {DB_PATH_1}")
    print(f"   - Part 2: {len(data_part2)}개 -> {DB_PATH_2}")

    # 3. 임베딩 모델
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 4. 함수 정의 (DB 생성용)
    def process_and_save(data_chunk, save_path, start_index_offset):
        texts = []
        ids = []
        metadatas = []
        
        # 데이터 가공
        for idx, item in enumerate(data_chunk):
            real_idx = start_index_offset + idx # 전체 기준 인덱스 (Lookup용)
            
            content = item.get('content', '').strip()
            source = item.get('source', '').strip()
            article = item.get('article', '').strip()
            
            if not content: continue
            
            full_text = f"[{source}] [{article}] {content}"
            texts.append(full_text)
            ids.append(str(real_idx)) # Lookup을 위해 원본 인덱스 저장
            metadatas.append({"source": source, "article": article})

        # 배치 처리 및 저장
        batch_size = 100
        first_batch = True
        vector_store = None
        
        print(f"👉 '{save_path}' 생성 중...")
        for i in tqdm(range(0, len(texts), batch_size), desc="   저장 중"):
            b_texts = texts[i : i+batch_size]
            b_ids = ids[i : i+batch_size]
            b_metas = metadatas[i : i+batch_size]
            
            if not b_texts: continue
            
            b_embeddings = embeddings.embed_documents(b_texts)
            
            if first_batch:
                vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=save_path,
                    collection_name="construction_laws"
                )
                first_batch = False
            
            vector_store._collection.add(
                ids=b_ids,
                embeddings=b_embeddings,
                metadatas=b_metas,
                documents=b_ids
            )
            time.sleep(0.5)

    # 5. 실행
    process_and_save(data_part1, DB_PATH_1, 0)          # 0번부터 시작
    process_and_save(data_part2, DB_PATH_2, mid_index)  # 중간번호부터 시작

    print(f"\n🎉 성공! 두 개의 폴더가 생성되었습니다.")
    print(f"1. {DB_PATH_1}")
    print(f"2. {DB_PATH_2}")
    print("이제 각각 70MB 정도일 겁니다. GitHub에 둘 다 올리세요!")

if __name__ == "__main__":
    create_split_vector_db()