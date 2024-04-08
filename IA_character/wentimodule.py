import random
import re

import emoji

from generate_vllm import generate2
from support_001 import generate, generateopenai

from support_001 import generate
with open('FilterWordxiaoyu22.txt', 'r', encoding='utf-8') as f:
    sensitive_words1 = [line.strip() for line in f.readlines()]
with open('FilterWordxiaoyu22.txt', 'r', encoding='utf-8') as f:
    sensitive_words2 = [line.strip() for line in f.readlines()]


def has_multiple_newlines(input_str):
    # 使用count函数来计算回车符的数量
    newline_count = input_str.count('\n')

    # 如果回车符数量大于1，说明有多个回车符
    if newline_count > 10:
        return True
    else:
        return False
def wentimodule(prompt,requirements,fc):
    mistake=0
    data = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': 200,
            'top_p': 0.75,
            'temperature': 0.95,
            'top_k': 40,
            'num_beams': 1,
            'repetition_penalty': 1.05
        }
    }
    wenti = generate(prompt,data)
    if wenti.find("方法:") != -1:
        fangfa = wenti.split("方法:")[-1]
        fangfa = fangfa.split("\n")[0]
        fangfa=fangfa.replace("回答:","")
        fangfa=fangfa.strip()
        wenti = wenti.split("方法:")[-1]
        if len(fangfa)>5:
            max_key = max(requirements.keys())
            new_key = max_key + 1
            fangfa = fangfa.strip()
            # 将新生成的问题字符串追加到 requirements 字典
            requirements[new_key] = fangfa
            with open(fc, 'w', encoding="utf-8") as file:
                for i in requirements:
                    file.write(requirements[i] + '\n')
            print("修正して方法を訂正します：", fangfa)
        print("修正して問題を訂正します：", wenti)
    if wenti.find("答案:")!=-1:
        wenti=wenti.split("答案:")[-1]
        print("修正して問題を訂正します：", wenti)
    if wenti.find("<br>")!=-1:
        wenti=wenti.split("<br>")[-1]
        print("修正して問題を訂正します：", wenti)
    if wenti.find("分析:") != -1:
        wenti = wenti.split("分析:")[-1]
        print("修正して問題を訂正します：", wenti)
    if wenti.find("回答:") != -1:
        wenti = wenti.split("回答:")[-1]
        print("修正して問題を訂正します：", wenti)


    if wenti.find("问题:") != -1:
        wenti = wenti.split("问题:")[-1]
        print("修正して問題を訂正します：", wenti)

    while any(word in wenti for word in sensitive_words1) or len(wenti)<6 or len(wenti)>500:# or has_multiple_newlines(wenti)==True
        print(f"error\nreloop")
        for word in sensitive_words1:
            if word in wenti:
                print(f"Found sensitive word: {word}")
        mistake+=1
        if mistake==3:
            break
        wenti = generate(prompt,data)
        if wenti.find("方法:") != -1:
            fangfa = wenti.split("方法:")[-1]
            fangfa = fangfa.split("\n")[-1]
            wenti = wenti.split("方法:")[-1]
            if len(fangfa) > 5:
                max_key = max(requirements.keys())
                new_key = max_key + 1
                fangfa = fangfa.strip()
                # 将新生成的问题字符串追加到 requirements 字典
                requirements[new_key] = fangfa
                with open(fc, 'w', encoding="utf-8") as file:
                    for i in requirements:
                        file.write(requirements[i] + '\n')
                print("修正して方法を訂正します：", fangfa)
            print("修正して問題を訂正します：", wenti)
        if wenti.find("答案:") != -1:
            wenti = wenti.split("答案:")[-1]
            print("修正して問題を訂正します：", wenti)
        if wenti.find("<br>") != -1:
            wenti = wenti.split("<br>")[-1]
            print("修正して問題を訂正します：", wenti)
        if wenti.find("分析:") != -1:
            wenti = wenti.split("分析:")[-1]
            print("修正して問題を訂正します：", wenti)
        if wenti.find("回答:") != -1:
            wenti = wenti.split("回答:")[-1]
            print("修正して問題を訂正します：", wenti)

    if mistake == 3:
        mistake=0
        return "error",requirements
    wenti=wenti.strip()
    return wenti,requirements
def has_long_consecutive_duplicates(input_string, threshold=10):
    consecutive_count = 1

    for i in range(len(input_string) - 1):
        if input_string[i] == input_string[i + 1]:
            consecutive_count += 1
            if consecutive_count >= threshold:
                return True
        else:
            consecutive_count = 1

    return ""
def has_emoji(input_str):
    #print(emoji.demojize(input_str).find(":"))
    # 定义正则表达式模式
    pattern = r":\w+:"

    # 使用 re.findall() 查找所有匹配的模式
    matches = re.findall(pattern, emoji.demojize(input_str))

    # 打印匹配的结果
    if matches:
        return True
    else:
        return False
def huidamodule(prompt):
    mistake=0
    data = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': 500,
            'top_p': 0.75,
            'temperature': 0.95,
            'top_k': 40,
            'num_beams': 1,
            'repetition_penalty': 1.05
        }
    }
    wenti = generate(prompt,data)

    if wenti.find("<br>")!=-1:
        wenti=wenti.split("<br>")[-1]
        print("修正して問題を訂正します：", wenti)
    if wenti.find("分析:") != -1:
        wenti = wenti.split("分析:")[-1]
        print("修正して問題を訂正します：", wenti)
    if wenti.find("方法:") != -1:
        wenti = wenti.split("方法:")[-1]
        print("修正して問題を訂正します：", wenti)
    if wenti.find("问题:") != -1:
        wenti = wenti.split("问题:")[-1]
        print("修正して問題を訂正します：", wenti)
    wenti=wenti.strip()
    while any(word in wenti for word in sensitive_words1) or len(wenti)<6 or len(wenti)>16666 or has_emoji(wenti)==False:
        print(f"error\nreloop",any(word in wenti for word in sensitive_words1))
        if has_emoji(wenti)==False:
            print(f"{wenti} 不包含表情符号。")
    # while any(word in wenti for word in sensitive_words1) or len(wenti)<6 or len(wenti)>500 or has_long_consecutive_duplicates(wenti, threshold=10)==True:
    #     print(f"error\nreloop",any(word in wenti for word in sensitive_words1),has_long_consecutive_duplicates(wenti, threshold=10))
        for word in sensitive_words1:
            if word in wenti:
                print(f"Found sensitive word: {word}")
        mistake+=1
        if mistake==3:
            break
        wenti = generate(prompt,data)
        if wenti.find("<br>")!=-1:
            wenti=wenti.split("<br>")[-1]
            print("修正して問題を訂正します：", wenti)
        if wenti.find("分析:") != -1:
            wenti = wenti.split("分析:")[-1]
            print("修正して問題を訂正します：", wenti)
        if wenti.find("方法:") != -1:
            wenti = wenti.split("方法:")[-1]
            print("修正して問題を訂正します：", wenti)
        if wenti.find("问题:") != -1:
            wenti = wenti.split("问题:")[-1]
            print("修正して問題を訂正します：", wenti)
        wenti = wenti.strip()
    if mistake == 3:
        mistake=0
        return "error"
    wenti=wenti.strip()
    wenti = wenti.replace("#方法#","")
    return wenti
def promptgen(prompt):
    mistake = 0
    data = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': 500,
            'top_p': 0.75,
            'temperature': 0.95,
            'top_k': 40,
            'num_beams': 1,
            'repetition_penalty': 1.05
        }
    }
    wenti = generate(prompt, data)
    wenti = wenti.strip()
    return wenti

def wentimodule_batch(stack1):
    mistake=0
    print(stack1[0])
    data = {
        "model": "Qwen1.5-14B-Chat_4bit2",
        "prompt": stack1,
        "stop_token_ids": [151645, 151644, 151643],
        "max_tokens": 4000,
        "top_p":0.9,
        "top_k":20,
        "temperature":0.95,
        "repetition_penalty":1.05,
        "do_sample":True,
    }  # 151645 <|im_end|> 151644 <|im_start|> 151643 <|endoftext|>

    ress = generate2(data)
    for i in range(len(ress)):
        wenti = ress[i]
        wenti = wenti.strip()
        wenti = wenti.replace("<|im_end|>", "")
        if "<br>" in wenti:
            wenti = wenti.split("<br>")[-1]
            print("修正并订正问题：", wenti)
        if "分析:" in wenti:
            wenti = wenti.split("分析:")[-1]
            print("修正并订正问题：", wenti)
        if "方法:" in wenti:
            wenti = wenti.split("方法:")[-1]
            print("修正并订正问题：", wenti)
        if "问题:" in wenti:
            wenti = wenti.split("问题:")[-1]
            print("修正并订正问题：", wenti)
        if "问题：" in wenti:
            wenti = wenti.split("问题：")[-1]
            print("修正并订正问题：", wenti)
        if "#" in wenti:
            wenti = wenti.split("#")[0]
            print("修正并订正问题：", wenti)

        # 将修改后的字符串重新赋值给列表中相应的位置
        ress[i] = wenti
    wentis=ress
    print(wentis[0])
    #wentis=[]
    # # 示例的问答对列表
    # # 将列表 a 和 b 中的元素组成问答对
    # for i in range(min(len(stack1), len(ress))):
    #     question = stack1[i]
    #     answer_b = ress[i]
    #     # 过滤掉包含敏感词或长度不符合条件的答案
    #     # if all(word not in answer_a for word in sensitive_words1) and 30 <= len(answer_a) <= 2048 and \
    #     #         all(word not in answer_b for word in sensitive_words1) and 30 <= len(answer_b) <= 2048:
    #     if all(word not in answer_b for word in sensitive_words1) and 30 <= len(answer_b) <= 2048:
    #         wentis.append(answer_b)


    return wentis
def huidamodule_batch(stack1):
    mistake = 0
    print(stack1[0])
    data = {
        "model": "Qwen1.5-14B-Chat_4bit2",
        "prompt": stack1,
        "stop_token_ids": [151645, 151644, 151643],
        "max_tokens": 4000,
        "top_p": 0.9,
        "top_k": 20,
        "temperature": 0.95,
        "repetition_penalty": 1.05,
        "do_sample": True,
    }  # 151645 <|im_end|> 151644 <|im_start|> 151643 <|endoftext|>

    ress = generate2(data)
    for i in range(len(ress)):
        wenti = ress[i]
        wenti = wenti.strip()
        wenti = wenti.replace("<|im_end|>", "")
        if "问题:" in wenti:
            wenti = wenti.split("问题:")[1]
            print("修正并订正问题：", wenti)
        if "问题：" in wenti:
            wenti = wenti.split("问题：")[1]
            print("修正并订正问题：", wenti)
        # 将修改后的字符串重新赋值给列表中相应的位置
        ress[i] = wenti
    huidas = ress
    print(huidas[0])
    # huidas = []
    # indices_to_remove = []
    #
    # for i in range(len(ress) - 1, -1, -1):
    #     answer_b = ress[i]  # Get the corresponding element from ress
    #     if not (all(word not in answer_b for word in sensitive_words1) or 30 <= len(answer_b) <= 2048):
    #         indices_to_remove.append(i)
    #     else:
    #         huidas.append(answer_b)
    #
    # for index in indices_to_remove:
    #     del stack1[index]
    #     del ress[index]

    return stack1,huidas

