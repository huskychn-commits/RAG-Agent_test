import os
import sys
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import socket

# 添加当前目录到sys.path以支持模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入封装好的函数
import Find_BaseDir
import remove_empty_lines
import fix_line_breaks

if __name__ == "__main__":
    # 依次执行三个脚本的核心功能
    print("\033[38;5;208m运行中: Find_BaseDir\033[0m")
    Find_BaseDir.update_env_file()
    print("\033[38;5;208m运行中: remove_empty_lines\033[0m")
    remove_empty_lines.database_RemoveEmptyLines()
    print("\033[38;5;208m运行中: fix_line_breaks\033[0m")
    fix_line_breaks.database_FixLineBreaks()
    print("\033[38;5;208m运行中: remove_empty_lines\033[0m")
    remove_empty_lines.database_RemoveEmptyLines()
    print("\033[36m所有处理完成！\033[0m")