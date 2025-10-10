import os
from PyPDF2 import PdfReader

def get_user_input():
    try:
        page_length = int(input("请输入单个batch的页数长度（默认11）: ") or "11")
        overlap_length = int(input("请输入batch之间重叠页数长度（默认1）: ") or "1")
        return page_length, overlap_length
    except ValueError:
        print("输入错误，将使用默认值：单文件长度11，重叠长度1")
        return 11, 1

def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    print(f"[调试] 输出目录已创建: {output_path}")

def process_pdf(pdf_path, page_length, overlap_length, output_path):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"[调试] 总页数: {total_pages}")
    
    batches = []
    start_page = 0
    
    while start_page < total_pages:
        end_page = min(start_page + page_length, total_pages)
        batches.append((start_page, end_page))
        print(f"[调试] Batch {len(batches)}: 页码范围 {start_page}-{end_page-1}")
        
        start_page = end_page - overlap_length
    
    for i, (start, end) in enumerate(batches, 1):
        output_file = os.path.join(output_path, f"batch_{i}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for page_num in range(start, end):
                text = reader.pages[page_num].extract_text()
                f.write(text + "\\n\\n")
        
        print(f"[调试] 已保存: {output_file}")

def main():
    # 假设PDF文件名为"book.pdf"，需要用户自行替换
    pdf_path = "数据库-哈利波特/book.pdf"
    output_path = ".aux/数据库-哈利波特/txt_batches"
    
    print("[调试] 开始处理PDF文件")
    page_length, overlap_length = get_user_input()
    create_output_dir(output_path)
    process_pdf(pdf_path, page_length, overlap_length, output_path)
    print("[调试] 处理完成")

if __name__ == "__main__":
    main()
