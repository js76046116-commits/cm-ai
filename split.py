from pypdf import PdfReader, PdfWriter
import os

def split_pdf():
    # 1. 파일 주소 입력 받기
    raw_path = input("분할할 PDF 파일의 전체 경로를 입력하세요: ").strip()
    input_path = raw_path.replace('"', '').replace("'", "")
    
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return

    # [수정] 페이지 수를 먼저 읽어서 알려줍니다.
    try:
        reader = PdfReader(input_path)
        total_pages = len(reader.pages)
        print(f"\n📄 확인된 파일: {os.path.basename(input_path)}")
        print(f"📊 총 페이지 수: {total_pages}페이지")
        print("-" * 40)
    except Exception as e:
        print(f"❌ PDF 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 2. 나눌 페이지 수 입력 받기 (총 페이지 확인 후 입력)
    try:
        chunk_size = int(input(f"몇 페이지씩 나누고 싶나요? (1 ~ {total_pages} 사이 숫자 입력): "))
        if chunk_size <= 0:
            print("❌ 1 이상의 숫자를 입력해주세요.")
            return
    except ValueError:
        print("❌ 숫자만 입력해주세요.")
        return

    # 저장 경로 설정 (원본 파일 폴더 내 split_results)
    source_dir = os.path.dirname(input_path)
    output_dir = os.path.join(source_dir, "split_results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"\n🔄 분할 작업을 시작합니다...")

    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        end = min(start + chunk_size, total_pages)
        
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        
        output_filename = f"{file_name}_part_{start//chunk_size + 1}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "wb") as f:
            writer.write(f)
            
        print(f"✅ 저장 완료: {output_filename} ({start+1}~{end}페이지)")

    print(f"\n✨ 작업 완료! 모든 파일이 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    split_pdf()