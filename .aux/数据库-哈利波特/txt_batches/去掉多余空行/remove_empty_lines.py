import os
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('txt_base_dir', "D:/课程作业/2025秋/人工智能/HW2/数据库-哈利波特/txt_batches")
def database_RemoveEmptyLines(base_dir=base_dir):
    n = 1
    
    while True:
        filename = os.path.join(base_dir, f'batch_{n}.txt')
        
        if not os.path.exists(filename):
            print(f"\033[91m未检测到文件：\033[0m{filename}")
            break
            
        print(f"\033[92m正在处理文件：\033[0m{filename}")
        
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read().replace('\\n', '\n')
        
        lines = [line + '\n' for line in content.split('\n') if line.strip()]
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.writelines(lines)
            
        n += 1

if __name__ == '__main__':
    database_RemoveEmptyLines(base_dir)
    print("\033[94m所有文件的多余空行已删除！\033[0m")
