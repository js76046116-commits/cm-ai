from pypdf import PdfWriter
import os

def merge_pdfs():
    # 1. PDF들이 들어있는 폴더 경로 입력 받기
    raw_path = input("합칠 PDF 파일들이 들어있는 폴더 경로를 입력하세요: ").strip()
    folder_path = raw_path.replace('"', '').replace("'", "")
    
    # 마지막 경로(폴더 이름) 추출
    # 예: C:/Users/Desktop/의정부_건축 -> '의정부_건축'
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    if not os.path.isdir(folder_path):
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return

    # 2. 폴더 내의 모든 PDF 파일 목록 가져오기 (이름순 정렬)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    pdf_files.sort()

    if not pdf_files:
        print("❌ 폴더 내에 PDF 파일이 없습니다.")
        return

    print(f"\n📂 대상 폴더: {folder_name}")
    print(f"📄 발견된 PDF 파일 ({len(pdf_files)}개):")
    for f in pdf_files:
        print(f" - {f}")
    print("-" * 40)

    # 3. 병합 작업 시작
    writer = PdfWriter()
    
    try:
        for pdf in pdf_files:
            file_path = os.path.join(folder_path, pdf)
            writer.append(file_path)
            print(f"➕ 추가 중: {pdf}")

        # [수정] 결과 파일명을 폴더 이름으로 설정
        output_filename = f"{folder_name}_merged.pdf"
        output_path = os.path.join(folder_path, output_filename)
        
        with open(output_path, "wb") as f:
            writer.write(f)
            
        print(f"\n✨ 병합 완료!")
        print(f"✅ 결과 파일: {output_path}")
        
    except Exception as e:
        print(f"❌ 병합 중 오류 발생: {e}")

if __name__ == "__main__":
    merge_pdfs()