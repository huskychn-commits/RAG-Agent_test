import os
import importlib.util
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import socket

# 动态加载哈利波特工具模块（如果存在）
faiss_module = None
question_to_context = None
try:
    faiss_module_path = os.path.abspath('.aux/数据库-哈利波特/harrypotter_agent_FAISS.py')
    if os.path.exists(faiss_module_path):
        spec = importlib.util.spec_from_file_location("harrypotter_agent_FAISS", faiss_module_path)
        faiss_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(faiss_module)
        question_to_context = faiss_module.question_to_context
        print("\033[94m哈利波特搜索工具已加载\033[0m")
    else:
        print("\033[93m警告：哈利波特搜索工具模块不存在，将跳过相关功能\033[0m")
except Exception as e:
    print(f"\033[93m警告：加载哈利波特工具失败: {e}\033[0m")

# 定义哈利波特搜索工具
tool = {
    "type": "function",
    "function": {
        "name": "search_harry_potter_context",
        "description": "获取与哈利波特相关的问题背景信息",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户提出的哈利波特相关问题"
                }
            },
            "required": ["question"]
        },
        "strict": True
    }
}
# 对于deepseek模型，tool_choice应该使用"none"或"auto"
tool_choice = "auto"

def is_connected_to_Internet(host="8.8.8.8", port=53, timeout=3):
    """
    尝试连接到 Google DNS（8.8.8.8:53），判断是否联网。
    可自定义 host/port。
    """
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

# ===== 可调参数配置 =====
MAX_RETRIES = 3                    # API重试次数
RETRY_DELAY = 5                    # API重试间隔（秒）
Internet_timeout = 10              # 网络超时时间（秒）
MAX_QUERY = 4                      # 最大查询文本数量
INITIAL_QUERY_COUNT = 1            # 初始查询文本数量
QUERY_INCREMENT = 1                # 每次查询增加的文本数量
STREAM_TIMEOUT = 3.0               # 流式传输超时时间（秒）

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
)

api_key = os.getenv("API_KEY")

messages = [
    {'role': 'system', 'content': '接下来你的每一个回答都不能是空的。\n\n当用户输入退出对话的指令（如"exit"）时，模型应返回固定结束语："Exiting the chat. Goodbye!"（不带引号）'}
]

while True:
    # Handle input with correct newline detection
    input_buffer = []
    last_empty = False
    
    print("\033[92mUser:\033[0m")  # Single prompt at start
    while True:
        current_input = input().strip()  # No prompt here
        
        if current_input == "":
            # Send on first empty line after input
            if not last_empty and input_buffer:  # First empty line after content
                user_input = "\n".join(input_buffer)
                input_buffer = []
                last_empty = True  # Keep state for consecutive empties
                break
            last_empty = True  # Maintain empty state
        else:
            input_buffer.append(current_input)
            last_empty = False  # Reset state on non-empty input
    
    # 初始化哈利波特工具使用标志
    harry_potter_used = False
    
    # 内部判断是否需要调用哈利波特工具
    if faiss_module and question_to_context:
        try:
            # 创建系统提示让LLM判断是否需要调用哈利波特工具
            system_prompt = {
                "role": "system",
                "content": "请分析用户的问题，判断是否与哈利波特相关。如果问题涉及哈利波特、魔法、霍格沃茨、伏地魔、魔法石、魁地奇等哈利波特相关主题，请返回YES，否则返回NO。只返回YES或NO，不要解释。"
            }
            
            # 内部调用LLM判断
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[system_prompt, {'role': 'user', 'content': user_input}],
                max_tokens=10,
                stream=False
            )
            
            # 解析判断结果
            judgment = response.choices[0].message.content.strip().upper()
            if judgment == "YES":
                print("\033[94m检测到哈利波特相关问题，将调用搜索工具...\033[0m")
                harry_potter_used = True
                
                # 让LLM思考需要哪些关键词
                keyword_prompt = {
                    "role": "system",
                    "content": f"用户的问题是：{user_input}\n\n请思考：要回答这个问题，需要从哈利波特原文中查找哪些信息？注意：原始文献仅包含哈利波特系列小说的原文内容，不包含外部知识。请提取最相关的关键词，这些关键词应该简洁且直接与哈利波特原文内容相关。请只返回关键词，用空格连接，不要解释。\n\n示例：\n- 问题：哈利波特今年几岁了？\n- 关键词：哈利波特 出生\n- 问题：赫敏的魔杖是什么材质的？\n- 关键词：赫敏 魔杖 材质\n- 问题：伏地魔有几个魂器？\n- 关键词：伏地魔 魂器 数量"
                }
                
                keyword_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[keyword_prompt],
                    max_tokens=50,
                    stream=False
                )
                
                search_keywords = keyword_response.choices[0].message.content.strip()
                print(f"\033[94m提取的关键词：{search_keywords}\033[0m")
                
                # 改进的反复核验机制：每次查询都让LLM提取相关语句
                extracted_contexts_list = []  # 存储每个查询轮次提取的相关语句列表
                current_query_count = INITIAL_QUERY_COUNT  # 初始查询文本数量
                sufficient_context_found = False
                
                while current_query_count <= MAX_QUERY and not sufficient_context_found:
                    # 使用提取的关键词调用哈利波特搜索工具，只获取当前查询数量的文本
                    current_contexts = question_to_context(search_keywords, top_batches=current_query_count)
                    if current_contexts:
                        print(f"\033[94m查询到{current_query_count}个文本，正在提取相关信息...\033[0m")
                        
                        # 让LLM从当前查询的文本中提取相关语句
                        extraction_prompt = {
                            "role": "system",
                            "content": f"用户的问题是：{user_input}\n\n关键词是：{search_keywords}\n\n以下是第{current_query_count}批查询到的文本信息：\n{'\n'.join(current_contexts)}\n\n请从这些文本中选出与关键词相关的语句（可以是和一个或几个关键词有关，只要他有可能有益于回答用户的问题）（可以是不同的几段）。如果文本中没有任何相关信息，请返回空列表。请只返回相关的语句，不要解释。"
                        }
                        
                        extraction_response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[extraction_prompt],
                            max_tokens=200,
                            stream=False
                        )
                        
                        extracted_text = extraction_response.choices[0].message.content.strip()
                        
                        # 将提取的语句添加到列表中（可以是空列表）
                        if extracted_text and extracted_text != "空列表" and extracted_text != "[]":
                            extracted_contexts_list.append(extracted_text)
                            print(f"\033[94m第{current_query_count}批文本提取到相关信息\033[0m")
                        else:
                            extracted_contexts_list.append("")  # 空列表
                            print(f"\033[94m第{current_query_count}批文本未提取到相关信息\033[0m")
                        
                        # 检查当前所有提取的信息是否足够
                        if extracted_contexts_list and any(extracted_contexts_list):  # 至少有一个非空列表
                            verification_prompt = {
                                "role": "system",
                                "content": f"以下是到目前为止提取的所有相关信息：\n{'\n'.join([ctx for ctx in extracted_contexts_list if ctx])}\n\n请判断你已有的知识加上这些信息是否足够回答用户的问题（“{user_input}”）？如果这些信息帮助你知道答案了，请返回YES；如果这些信息让你完全不知道怎么回答，请返回NO（否则都回答YES）。只返回YES或NO，不要解释。"
                            }
                            
                            verification_response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[verification_prompt],
                                max_tokens=10,
                                stream=False
                            )
                            
                            verification_result = verification_response.choices[0].message.content.strip().upper()
                            
                            if verification_result == "NO":
                                print(f"\033[94m查询到{current_query_count}个文本，信息不足，继续查询...\033[0m")
                                current_query_count += QUERY_INCREMENT  # 每次增加文本数量
                            else:
                                sufficient_context_found = True
                                print(f"\033[94m查询到{current_query_count}个文本，信息已足够\033[0m")
                        else:
                            # 如果没有任何提取的信息，继续查询
                            print(f"\033[94m查询到{current_query_count}个文本，未提取到相关信息，继续查询...\033[0m")
                            current_query_count += QUERY_INCREMENT
                    else:
                        break
                
                # 打包所有提取的信息
                all_extracted_contexts = [ctx for ctx in extracted_contexts_list if ctx]  # 过滤掉空列表
                
                if all_extracted_contexts:
                    if sufficient_context_found:
                        # 添加上下文到消息历史 - 使用所有提取的相关信息
                        messages.append({'role': 'system', 'content': '以下是与哈利波特相关的问题背景信息：\n' + '\n'.join(all_extracted_contexts)+'\n\n请依据这些信息回答用户的问题。'})
                    else:
                        # 达到最大查询次数但仍不足，但仍有提取的信息 - 添加黄色警告并显示提取的信息
                        print(f"\033[93m⚠️ 警告：查询了{MAX_QUERY}个文本仍无法获得足够信息，但找到以下相关信息：\033[0m")
                        for i, context in enumerate(all_extracted_contexts):
                            print(f"\033[90m提取信息{i+1}: {context[:100]}...\033[0m")
                        # 添加上下文到消息历史，但标记为可能不足
                        messages.append({'role': 'system', 'content': '以下是与哈利波特相关的问题背景信息：\n' + '\n'.join(all_extracted_contexts)+'\n\n注意：这些信息可能不足以完全回答用户的问题，请谨慎参考。如果信息不足，请基于你的知识补充回答。'})
                else:
                    # 完全没有提取到任何信息
                    print(f"\033[91m⚠️ 警告：查询了{MAX_QUERY}个文本未找到任何相关信息\033[0m")
                    messages.append({'role': 'system', 'content': '工具没有能查询到相关信息，需要你自行判断如何回答。'})
                
                # 添加系统提示，提醒LLM开始回答问题
                messages.append({'role': 'system', 'content': f'现在请直接回答用户的问题：{user_input}。不要使用"我来帮您查询"、"我来查找"等表明动作的语句。请基于你的知识直接给出答案，如果不知道答案，请直接说"我不知道"或"根据我的知识，这个问题没有明确的答案"。'})
                
                # 记录当前轮次添加的系统消息数量，用于后续清理
                current_round_system_messages = len([msg for msg in messages if msg['role'] == 'system']) - 1  # 减去初始的退出提示
        except Exception as e:
            print(f"\033[93m警告：判断哈利波特工具调用失败: {e}\033[0m")
    
    messages.append({'role': 'user', 'content': user_input})
    
    '''
    # 调试信息：显示是否决定调用哈利波特工具
    if harry_potter_used:
        print("\033[95m[调试] Agent决定调用哈利波特搜索工具\033[0m")
    else:
        print("\033[95m[调试] Agent决定不调用哈利波特搜索工具\033[0m")
    '''
    
    print("\033[93mAgent:\033[0m")  # Added yellow color to Agent
    
    retry_count = 0
    exit_flag = False
    while retry_count < MAX_RETRIES:
        try:
            # Handle API call with retry mechanism
            # 网络检查循环
            network_retries = 0
            while not is_connected_to_Internet():
                network_retries += 1
                if network_retries > MAX_RETRIES:
                    print(f"Error: Reached maximum retry limit ({MAX_RETRIES}). Please check network connection.")
                    retry_count=2
                    break
                print(f"Error: No Network connection. Retrying in {Internet_timeout} seconds, {network_retries}/{MAX_RETRIES}...")
                time.sleep(Internet_timeout)
            
            # 创建API调用参数 - 改回流式传输
            api_kwargs = {
                "model": "deepseek-chat",
                "messages": messages,
                "timeout": Internet_timeout,
                "stream": True  # 改回流式传输
            }
                
            # 添加工具调用支持
            if harry_potter_used:
                api_kwargs["tools"] = [tool]
                api_kwargs["tool_choice"] = tool_choice
                
            completion = client.chat.completions.create(**api_kwargs)
            
            response = ""
            tool_call = None
            
            # 处理流式响应 - 改进多行文本处理
            response = ""
            tool_call = None
            last_content_time = time.time()
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
                    print(content, end="", flush=True)  # Stream output
                    last_content_time = time.time()  # 更新最后内容时间
                    
                # 处理工具调用
                if hasattr(chunk.choices[0].delta, "tool_call") and chunk.choices[0].delta.tool_call:
                    tool_call = chunk.choices[0].delta.tool_call
                    last_content_time = time.time()  # 工具调用也更新计时器
            
            # 等待一小段时间确保流完全结束
            while time.time() - last_content_time < STREAM_TIMEOUT:
                time.sleep(0.1)
            
            print()  # Newline after streaming completes
            
            # 处理工具调用响应
            if tool_call and faiss_module:
                try:
                    # 执行工具调用
                    tool_response = faiss_module.search_harry_potter_context(**tool_call.function.arguments)
                    # 将工具调用和响应添加到消息历史
                    messages.append({'role': 'assistant', 'content': '', 'tool_call': tool_call})
                    messages.append({'role': 'tool', 'content': tool_response, 'name': tool_choice})
                    # 重新调用API获取最终回复 - 改回流式传输
                    completion = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        timeout=Internet_timeout,
                        stream=True  # 改回流式传输
                    )
                    response = ""
                    last_content_time = time.time()
                    
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            response += content
                            print(content, end="", flush=True)  # Stream output
                            last_content_time = time.time()  # 更新最后内容时间
                    
                    # 等待一小段时间确保流完全结束
                    while time.time() - last_content_time < STREAM_TIMEOUT:
                        time.sleep(0.1)
                    
                    print()  # Newline after streaming completes
                except Exception as e:
                    print(f"\033[93m警告：工具调用失败: {e}\033[0m")
            
            if not response:  # Handle empty response
                print(f"Error: Empty response from model. Retry {retry_count + 1}/{MAX_RETRIES}...")
                
                # 改进的空输出处理机制：清理最后的agent角色输出
                print("\033[93m清理最后的agent角色输出...\033[0m")
                
                # 找到并删除最后的assistant角色消息
                assistant_messages = [i for i, msg in enumerate(messages) if msg['role'] == 'assistant']
                if assistant_messages:
                    last_assistant_index = assistant_messages[-1]
                    # 删除最后的assistant消息
                    messages.pop(last_assistant_index)
                    print(f"\033[93m已删除最后的assistant消息（索引{last_assistant_index}）\033[0m")
                
                # 添加系统提示，要求LLM不要返回空输出
                retry_prompt = {
                    'role': 'system', 
                    'content': '请回答用户的问题，不要返回空输出。确保你的回答是有内容的，或者干脆表示你不知道。'
                }
                messages.append(retry_prompt)
                
                print("\033[93m重新调用LLM API...\033[0m")
                retry_count += 1
                import time
                time.sleep(RETRY_DELAY)
                continue  # 立即重新调用API
                
            messages.append({'role': 'assistant', 'content': response})
            
            # 参考资料分析环节 - 在清理缓存信息之前
            if harry_potter_used and 'all_extracted_contexts' in locals() and all_extracted_contexts and 'sufficient_context_found' in locals() and sufficient_context_found:
                try:
                    # 询问LLM是否参考了工具资料
                    reference_prompt = {
                        "role": "system",
                        "content": f"你刚才的回答是：{response}\n\n请分析：你的回答是否参考了之前提供的哈利波特资料？如果有，请具体说明参考了哪些部分的内容。如果没有参考，请说明。请详细分析。"
                    }
                    
                    reference_response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[reference_prompt],
                        max_tokens=200,
                        stream=False
                    )
                    
                    reference_analysis = reference_response.choices[0].message.content.strip()
                    
                    # 显示参考资料分析（紫色标题，灰色内容）
                    print("\033[95m参考资料：\033[0m")
                    print(f"\033[90m{reference_analysis}\033[0m")
                    print()  # 空行分隔
                    
                except Exception as e:
                    print(f"\033[93m警告：参考资料分析失败: {e}\033[0m")
            
            # 如果使用了哈利波特工具，清理当前轮次添加的系统消息
            if harry_potter_used and 'current_round_system_messages' in locals():
                # 保留初始的退出提示和用户/助手对话，删除当前轮次的系统消息
                messages_to_keep = [messages[0]]  # 保留初始退出提示
                messages_to_keep.extend([msg for msg in messages if msg['role'] in ['user', 'assistant']])
                messages = messages_to_keep
            
            # 清理重试过程中添加的系统提示（如果有）
            # 只保留初始退出提示和用户/助手对话
            final_messages = [messages[0]]  # 保留初始退出提示
            final_messages.extend([msg for msg in messages if msg['role'] in ['user', 'assistant']])
            messages = final_messages
            
            if response.strip() == "Exiting the chat. Goodbye!":
                exit_flag = True
                break
            break  # Success, exit retry loop
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down gracefully...")
            break
        except Exception as e:
            print(f"\033[95m[调试] LLM返回错误: {str(e)}\033[0m")
            print(f"Error: {str(e)}. Retry {retry_count + 1}/{MAX_RETRIES}...")
            retry_count += 1
            time.sleep(RETRY_DELAY)
            continue

    
    if exit_flag:
        break

    if retry_count >= MAX_RETRIES:
        print(f"Error: Reached maximum retry limit ({MAX_RETRIES}).")
        #break  # Continue to next iteration of main loop

# Save messages to file after program execution

base_dir = os.getenv("BASE_DIR", "default_conversation_dir")
os.makedirs(base_dir, exist_ok=True)
current_time = time.strftime("%Y%m%d_%H%M%S")
filename = os.path.join(base_dir, f"{current_time}_conversation.txt")

with open(filename, "w", encoding="utf-8") as f:
    for message in messages:
        if message['role'] in ['user', 'assistant']:
            f.write(f"{message['role']}:\n {message['content']}\n\n")
print(f"\033[94mConversation saved to: {filename}\033[0m")
