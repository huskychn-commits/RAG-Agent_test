import os
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import socket

def database_FixLineBreaks(base_dir=None):
    
    def fix_line_breaks(text):
        """通过API调用清理多余换行符"""
        
        # 内部网络检查函数
        def is_connected_to_Internet(host="8.8.8.8", port=53, timeout=3):
            try:
                socket.setdefaulttimeout(timeout)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                s.close()
                return True
            except Exception:
                return False
                
        # 配置参数
        MAX_RETRIES = 3
        RETRY_DELAY = 2
        Internet_timeout = 5  # 网络超时时间
        
        # 初始化API客户端
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 系统提示词
        messages = [
            {'role': 'system', 'content': '请分析用户提供的文本并移除所有多余的换行符。保持原始文本内容不变，仅去除不必要的换行（即影响语义通顺或语法格式或通常习惯的换行应当被去掉，而可去可不去的换行应当保留）。返回清理后的文本结果，只需返回结果文本，不能包含其他内容；如果转换失败或清理后文本为空，不要返回任何内容，直接返回换行符\\n'}
        ]
        
        # 添加用户输入
        messages.append({'role': 'user', 'content': text})
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # 网络检查循环
                network_retries = 0
                while not is_connected_to_Internet():
                    network_retries += 1
                    if network_retries > MAX_RETRIES:
                        retry_count=2
                        raise Exception(f"网络连接失败，已达到最大重试次数({MAX_RETRIES})")
                    print(f"\033[93m错误: \033[0m无网络连接。\033[93m正在尝试第 {network_retries}/{MAX_RETRIES} 次重试...\033[0m")
                    time.sleep(Internet_timeout)
                
                completion = client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    stream=False
                )
                
                response = completion.choices[0].message.content
                
                if not response:
                    raise Exception(f"API返回空响应")
                
                '''
                if response.strip() == "\\n":
                    return None
                '''
                return response
                
            except Exception as e:
                print(f"\033[93m错误: \033[0m{str(e)}。\033[93m第 {retry_count + 1}/{MAX_RETRIES} 次重试...\033[0m")
                retry_count += 1
                time.sleep(RETRY_DELAY)
                
        return None

    """处理指定目录下的文本文件，修复换行符问题"""
    if base_dir is None:
        base_dir = os.getenv('txt_base_dir', "D:/课程作业/2025秋/人工智能/HW2/数据库-哈利波特/txt_batches")
    
    n = 1
    while True:
        filename = os.path.join(base_dir, f'batch_{n}.txt')
        
        if not os.path.exists(filename):
            print(f"\033[91m未检测到文件：\033[0m{filename}")
            break
            
        print(f"\033[94m正在处理文件：\033[0m{filename}")
        
        # 读取文件内容
        with open(filename, 'r', encoding='utf-8') as file:
            input_text = file.read()
        
        # 调用API修复换行符
        result = fix_line_breaks(input_text)
        
        if result:
            # 写回清理后的内容
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(result)
            print(f"\033[92m已修复换行符：\033[0mbatch_{n}.txt")
        else:
            print(f"\033[91m处理失败：\033[0mbatch_{n}.txt")
            
        n += 1

if __name__ == '__main__':
    base_dir = os.getenv('txt_base_dir', "D:/课程作业/2025秋/人工智能/HW2/数据库-哈利波特/txt_batches")
    database_FixLineBreaks(base_dir)
    print("\033[92m所有文件的换行符已修复！\033[0m")
