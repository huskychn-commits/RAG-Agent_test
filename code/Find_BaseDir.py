import os
import re

def update_env_file():
    # 获取脚本自身路径并计算目标路径
    script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    
    # 读取.env文件内容
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 构建正则表达式（匹配变量定义，允许=号周围有空格，保留引号，支持正斜线路径）
    pattern = r'base_dir\s*=\s*"[^"]*"'
    
    # 替换现有变量或追加新变量
    if re.search(pattern, content):
        # 替换现有变量（转换路径为正斜线格式）
        normalized_path = base_dir.replace('\\', '/')
        content = re.sub(pattern, f'base_dir = "{normalized_path}"', content)
    else:
        # 追加新变量（保持与文件现有格式一致）
        content += f'\nbase_dir = "{base_dir}"\n'
    
    # 写回.env文件
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\033[94m已更新 .env 文件中的 base_dir 变量为：\033[0m{base_dir}")

if __name__ == '__main__':
    update_env_file()
