import os
import re

def update_env_file():
    # 获取脚本所在目录
    pkl_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    
    # 读取.env文件内容
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 构建正则表达式（匹配变量定义，允许=号周围有空格，保留引号，支持正斜线路径）
    pattern = r'pkl_base_dir\s*=\s*"[^"]*"'
    
    # 替换现有变量或追加新变量
    if re.search(pattern, content):
        # 替换现有变量（转换路径为正斜线格式）
        normalized_path = pkl_base_dir.replace('\\', '/')
        content = re.sub(pattern, f'pkl_base_dir = "{normalized_path}"', content)
    else:
        # 追加新变量（保持与文件现有格式一致）
        content += f'\npkl_base_dir = "{pkl_base_dir}"\n'
    
    # 写回.env文件
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\033[94m已更新 .env 文件中的 pkl_base_dir 变量为：\033[0m{pkl_base_dir}")

    # 更新config.yaml的input_dir和output_dir
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # 修改input_dir路径
    input_pattern = r'input_dir:\s*"[^"]*"'
    normalized_path = pkl_base_dir.replace('\\', '/')
    config_content = re.sub(input_pattern, f'input_dir: "{normalized_path}/txt_batches"', config_content)
    
    # 修改output_dir路径
    output_pattern = r'output_dir:\s*"[^"]*"'
    config_content = re.sub(output_pattern, f'output_dir: "{normalized_path}/embedding/output"', config_content)
    
    # 写回config.yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\033[94m已更新 config.yaml 的 input_dir 和 output_dir 路径\033[0m")

if __name__ == '__main__':
    update_env_file()
