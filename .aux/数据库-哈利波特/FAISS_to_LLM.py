from harrypotter_agent_FAISS import HarryPotterAgent

def question_to_context(question, top_batches=3):
    """将问题转换为上下文文本列表"""
    agent = HarryPotterAgent()  # 创建HarryPotterAgent实例
    batch_results = agent.find_relevant_batches(question, top_batches)  # 调用新方法
    file_paths = [result[0] for result in batch_results]  # 提取文件路径
    contexts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            contexts.append(f.read())
    return contexts

if __name__ == "__main__":
    # 示例使用
    sample_question = "哈利波特的父亲是谁"
    results = question_to_context(sample_question)
    
    # 输出前50个字符
    for i, text in enumerate(results):
        print(f"段落{i+1}前50字符: \n{text[:50]}\n")
