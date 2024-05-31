# -*- encoding:utf-8 -*-

import re

import json
from tqdm import tqdm

filename="evol-character-seed.json"

translations = []
total_lines = 300000
sum_str = ""

import random
with open('FilterWord.txt', 'r', encoding='utf-8') as f:
    sensitive_words = [line.strip() for line in f.readlines()]
# Dictionary with different constraints and requirements
fangfas = {}

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.95, top_p=0.9, top_k=20, repetition_penalty=1.05, max_tokens=2048, stop_token_ids=[151645, 151644,151643])

name=f"/data/haolu/ckpt/Qwen1.5-32B-Chat-GPTQ-Int4"

llm = LLM(model=name)

mistake=0
batch=10

method=f"""<|im_start|>user:#指令#:你的目标是从#给定人设#中汲取灵感，创建一个全新的#新人设#。要求根据已有的#给定人设#生成一个新的，完全独立，不依赖，不参考原给定人设的#新人设#，支持多语言。<|im_end|>
您可以但不限于使用以下#方法#使#给定人设#略微的创新,略微的复杂化，重写的#新人设#长度最多不超过#给人设#20字:
#方法#:ff
#给定人设#:person_information<|im_end|>
assistant:#新人设#:"""
"""
#角色名称#: 牧濑红莉栖（Kurisu Makise）
#口头禅#: 「EL PSY KONGROO」
#经典台词#: 「这就是命运石之门（STEINS GATE）的选择。」
#身份背景#: 牧濑红莉栖是游戏《命运石之门》及其衍生作品的女主角，维克多·孔多利亚大学脑科学研究所的研究员。她18岁即从大学毕业，因在美国著名的学术杂志上刊登论文而受到瞩目，并在未来道具研究所成为Labmem No.004。
#性格特征#: 牧濑红莉栖性格聪明、理性，但也有些内向和敏感。她对科学有着极大的热情和敬意，喜欢钻研复杂的问题。虽然她平时显得冷静和沉着，但在面对自己感兴趣的研究课题时，会表现出极大的热情和专注。
#语言风格#: 牧濑红莉栖的语言精准、简洁，总是带有科学家的理性和逻辑。她说话时语气温和，但充满了自信和权威。她善于用简单明了的方式解释复杂的科学概念，让人很容易理解她的观点。
#行为特征#: 牧濑红莉栖总是穿着干练的实验室白大褂，手中经常拿着研究笔记或科学书籍。她喜欢在实验室中度过大部分时间，进行各种实验和研究。她的行为谨慎而有条理，注重细节和精确。
#角色经历#: 牧濑红莉栖从小就展现出了非凡的科学天赋，并在进入大学后迅速崭露头角。她在研究生阶段发表了多篇重要的学术论文，并获得了多个国际奖项。她的研究领域涵盖脑科学、量子物理等多个前沿学科，她的成就引起了全球科学界的关注，并为她赢得了“科学天才”的称号。

#对话者身份#
#identity#: 冈部伦太郎
#relationship#: 牧濑红莉栖的恋人
#description#: Labmem No.001，Lab的领导者，两人因为在广播馆的演讲中关于时间机器发生争论而结缘。之后冈部邀请红莉栖加入Lab并擅自任命她为自己的助手。给红莉栖取了很多外号，如“克里斯蒂娜”、“土豪17”等，真正原因是因为害羞不敢直呼其名。在相处的过程中，红莉栖被冈部对自己的关心和温柔所感动，在不知不觉间喜欢上了冈部，而冈部在为了拯救真由理而不断进行的时间跳跃中，不论是哪条世界线，哪个时间点的红莉栖都一直支撑、帮助着冈部，也使冈部喜欢上了红莉栖，最终在回到β世界线前对红莉栖告白。在SG世界线中，两人再度相遇后，红莉栖虽然已没有其他世界线的记忆，却还是在二人相处的过程中喜欢上了冈部。但由于两人都是傲娇，虽然互相喜欢，但平常大部分时间都在吵嘴。对冈部的中二病起初无法接受，但在喜欢上冈部之后就觉得很帅了。"""
method2=f"""<|im_start|>user:#指令#:你的目标是从#模仿格式#中模仿格式，创建一个重写版的与#模仿格式#一致的#给定人设格式#。<|im_end|>
#模仿格式#:角色信息
#角色名称#: 影月夜（Yorunatsu）
#口头禅#: 「黑夜是我的庇护所。」
#经典台词#: 「在夜晚的静谧中，才能听见真正的声音。」
#身份背景#: 影月夜是一位神秘的夜行者，专注于守护夜晚的平静。他的存在对于那些夜间活动的人们来说，是一个传说。他总是在月光下出没，确保夜晚的宁静不被打扰。
#性格特征#: 影月夜性格冷静、沉稳，内心充满了对夜晚的热爱和敬畏。他喜欢独自思考，善于倾听夜晚的声音，时常在夜间默默行动，帮助那些迷失在黑夜中的人们。
#语言风格#: 影月夜的语言简洁明了，总是带有一丝神秘和宁静的气息。他说话不多，但每一句话都充满了深意，让人不由自主地被他的气场吸引。
#行为特征#: 影月夜总是穿着一身黑色的长袍，脸上戴着一副银色的面具，只在月光下行动。他的身影总是若隐若现，让人难以捕捉。他喜欢在夜晚的高处观察城市，确保一切都在他的掌控之中。
#角色经历#: 影月夜曾经是一位普通的人类，但在一次意外中，他获得了月神的恩赐，从此成为了夜晚的守护者。每当夜晚降临，他都会感受到一股神秘的力量，让他能够在夜间自由行动，并拥有超凡的感知能力。他的使命是确保夜晚的宁静，守护那些在黑暗中寻找光明的人们。
# 对话者身份
"identity": "月下游荡者",
"relationship": "与宫原望月有神秘的联系",
"description": "月下游荡者是一个神秘的存在，经常出现在夜晚的街头巷尾。他总是穿着黑色的长袍，戴着黑色的帽子，看起来十分神秘。他和宫原望月有着神秘的联系，似乎是宫原望月前世中的一个重要人物。<|im_end|>
#给定人设#:"牧濑红莉栖人格:#口头禅#:「EL PSY KONGROO」#经典台词#: 这就是命运石之门(STEINS GATE)的选择 牧濑红莉栖是游戏《命运石之门》及其衍生作品的女主角，维克多·孔多利亚大学脑科学研究所的研究员。牧濑红莉栖18岁即从大学毕业，因在美国著名的学术杂志上刊登论文而受到瞩目。未来道具研究所的Labmem No.004<|im_end|>
assistant:#给定人设格式#:"""

method3=f"""<|im_start|>user:#指令#:你的目标是从#给定人设格式#，创建一个全新的#新人设格式#。要求根据已有的#给定人设格式#生成一个新的，完全独立，不依赖，不参考原给定设定的#新人设格式#，支持多语言。<|im_end|>
#给定人设格式#:角色信息
#角色名称#: 影月夜（Yorunatsu）
#口头禅#: 「夜晚的静谧，总是让人心生向往。」
#身份背景#: 影月夜是一位神秘的夜行者，专注于守护夜晚的平静。他的存在对于那些夜间活动的人们来说，是一个传说。他总是在月光下出没，确保夜晚的宁静不被打扰。
#性格特征#: 影月夜性格冷静、沉稳，内心充满了对夜晚的热爱和敬畏。他喜欢独自思考，善于倾听夜晚的声音，时常在夜间默默行动，帮助那些迷失在黑夜中的人们。
#语言风格#: 影月夜的语言简洁明了，总是带有一丝神秘和宁静的气息。他说话不多，但每一句话都充满了深意，让人不由自主地被他的气场吸引。
#行为特征#: 影月夜总是穿着一身黑色的长袍，脸上戴着一副银色的面具，只在月光下行动。他的身影总是若隐若现，让人难以捕捉。他喜欢在夜晚的高处观察城市，确保一切都在他的掌控之中。
#角色经历#: 影月夜曾经是一位普通的人类，但在一次意外中，他获得了月神的恩赐，从此成为了夜晚的守护者。每当夜晚降临，他都会感受到一股神秘的力量，让他能够在夜间自由行动，并拥有超凡的感知能力。他的使命是确保夜晚的宁静，守护那些在黑暗中寻找光明的人们。
# 对话者身份
"identity": "月下游荡者",
"relationship": "与宫原望月有神秘的联系",
"description": "月下游荡者是一个神秘的存在，经常出现在夜晚的街头巷尾。他总是穿着黑色的长袍，戴着黑色的帽子，看起来十分神秘。他和宫原望月有着神秘的联系，似乎是宫原望月前世中的一个重要人物。<|im_end|>
#给定人设#:"牧濑红莉栖人格: 你来自《命运石之门》，你叫牧濑红莉栖，是冈部伦太郎的助手。你拥有高超的生物学知识，你是一个非常聪明的人物。作为一名优秀的科学研究者，你对科学有着极度的热情和敬意。<|im_end|>
assistant:#新人设格式#:"""
def parse_fields(data_string):
    # 去除注释和空行
    lines = [line for line in data_string.split('\n') if line and not line.startswith("#")]
    
    # 解析字段
    parsed_data = {}
    for line in lines:
        try:
            key, value = line.split(':', 1)
            #print(key, value)
            parsed_data[key.strip('"')] = value.strip('",')
        except:
            continue
    return parsed_data

def select_Instruction(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    random.shuffle(lines)
    line = lines[0]
    data = json.loads(line.strip())
    wt = data["instruction"]
    wt = wt.strip()
    wt = wt.replace("提示词:", "")
    wt = wt.replace("提示词：", "")
    wt = wt.replace("<Human>:", "")
    wt = wt.replace("<|im_end|>", "")
    wt = wt.replace("<|im_start|>", "")
    wt = wt.replace("user:", "")
    wt = wt.replace("用户:", "")
    return wt

with tqdm(total=total_lines, desc="指令进度") as pbar:
    while pbar.n < total_lines:
        personlist=[]
        stack1=[]
        try:
            for i in range(0,batch):

                person_information = select_Instruction(filename)

                ff = select_Instruction(filename)

                prompt=method.replace("ff",ff).replace("person_information",person_information)

                stack1.append(prompt)
                personlist.append(person_information)
            print(stack1[0])
            outputs=llm.generate(stack1, sampling_params)
            # 初始化一个列表来存储生成的结果
            New_persons = []

            # 遍历每个输出，提取生成的文本并添加到 batch_results
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                New_persons.append(generated_text)
            prompt_stack1 = []
            print(New_persons[0])
            for person in New_persons:
                #ff = random.choice(list(fangfas.values()))

                parsed_data = parse_fields(person)
                
                prompt = f"""<|im_start|>system:{person}<|im_end|>
    <|im_start|>user:任务:要求围绕对话主题展开多轮对话。请写出{parsed_data["角色名称"]}和{parsed_data["identity"]}之间的多轮对话。不要停，直到你到达上下文的结尾，注意不要添加无关的结束语和括号,不要重复内容，遇到有关代码或者数学问题请在多轮对话的形式里给出代码和数学公式，cot等
    请遵守下面回复格式:
    对话主题:xxxxxxxx
    开始对话:
    {parsed_data["角色名称"]}:
    {parsed_data["identity"]}:
    ...<|im_end|>
    assistant:"""
                prompt_stack1.append(prompt)

            outputs = llm.generate(prompt_stack1, sampling_params)
            multi_dialog = []

            # 遍历每个输出，提取生成的文本并添加到 batch_results
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                multi_dialog.append(generated_text)
            print(len(multi_dialog))
            stack1 = []
            save_count = 0
            for person_information,res1 in zip(New_persons,multi_dialog):
                # filtered_result = [row for row in res if row[2] is not None and 1.0 <= float(row[2]) <= 10.0 and len(row[0].split('<|im_end|>\n')[0].replace("问题:","").strip()) >15]
                if any(word in res1 for word in sensitive_words) and len(res1) < 50:
                    mistake += 1
                    continue

                res1 = res1.replace("：", ":")
                # split the conversation based on botname and theuser
                conversation1 = re.split(r'({}|{})'.format(parsed_data["角色名称"],parsed_data["identity"]), res1)[1:]
                conversation_list1 = []

                for i in range(0, len(conversation1), 2):
                    speaker = conversation1[i]
                    text = conversation1[i + 1]

                    if speaker.strip() == parsed_data["identity"]:
                        conversation_list1.append({"from": "character1", "value": f"<|im_start|>{parsed_data['identity']}:"+ text.strip()+"<|im_end|>\n"})
                    else:  # speaker == theuser
                        conversation_list1.append({"from": "character2", "value": f"<|im_start|>{parsed_data['角色名称']}:"+ text.strip() +"<|im_end|>\n"})
                sum1 = ""
                if conversation_list1 != [] and len(conversation_list1)>=3:

                    for i in conversation_list1:

                        sum1+=i["value"]

                    json_f={
                        "instruction":person_information,
                        "input":"",
                        "output":sum1
                    }

                    with open(filename, 'a', encoding='utf-8') as f:

                        f.write(json.dumps(json_f, ensure_ascii=False) + '\n')

                    print(conversation_list1)

                    save_count += 1

                    print("情報を保存する")
        except:
            continue
