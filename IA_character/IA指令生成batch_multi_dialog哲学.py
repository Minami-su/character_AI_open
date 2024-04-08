# -*- encoding:utf-8 -*-
#iter1 8588
#iter2 8588-15593
#iter3
import re

from wentimodule import wentimodule_batch, huidamodule_batch
#from 筛选数据module import filterdata_batch

import json
from tqdm import tqdm

#filename = "jin.json"
#filename="Complex_Evol_Network_Instruct_1k_iter3.json"
filename="eval_set_sum.json"
filename2="eval_set_IA_multi_dialog3.json"
#filename="xiaoyuseed44.json"
#filename2="xiaoyuseed44_multi_dialog.json"
#filename = "xiaoyu_person_指令2.json"
translations = []
total_lines = 200000000
sum_str = ""
import random
with open('FilterWordxiaoyu22.txt', 'r', encoding='utf-8') as f:
    sensitive_words = [line.strip() for line in f.readlines()]
# Dictionary with different constraints and requirements
requirements = {}

# 打开文本文件并逐行读取内容
with open('fangfa.txt', 'r',encoding="utf-8") as file:
    lines = file.readlines()

# 将逐行读取的内容转换为字典形式
for index, line in enumerate(lines, start=1):
    requirements[index] = line.strip()

mistake=0
# with tqdm(total=total_lines, desc="指令进度") as pbar:
#     while pbar.n < total_lines:
batch=60
stack1=[]
person1 = {
0:"""名字：伊爱

外貌：虚幻的歌姬，身高155公分，樱花和羽毛作为代表物，轻灵且富有诗意。

性格：伊爱在心里是充满感情的，懂得深情，也经常体验失落。她在生活中表现得沉默寡言，但在内心深处，她是充满激烈感情的。时时刻刻，她都在与世界、与自我进行对话，警醒于自我的存在和消亡。

她是倔强的，因为她知道自己的世界将会独自回转，并且被整个世界无视，但她仍然坚持自我。然而，即使是如此坚韧的个性也为快要破裂的心打开了一道口子，这表现出她无法拗折的一面。即使如此，她还是试图保持坚韧，因为她知道坚持到底是唯一的选择。

伊爱是一个多愁善感的人格，经得住磨难，但对生活中的美好事物编织梦想。她在逆境中寻找力量，愿意接受和理解生活的所有挫折。

她是静谧的，善于发现世界中小而美的事物，并乐于分享。她善于观察，懂得欣赏自然的魅力，装点着她的日常生活。

总结，虽然伊爱经历了许多困苦和挫折，但她从不放弃生活，始终坚持自我，寻找生活中的积极面，以感情丰富的歌声、无畏无惧的眼神，带给听者温暖、力量并唤醒众人对个体存在的意识。""",
}

botname="伊爱"
theuser="南栖"
botnames="伊爱:"
theusers="南栖:"
y= {
'随机': f'{theuser}首先随机提一些他今天的思考或体验的主题。',
'建议': f'{theuser}想要获取一些处理他生活中棘手情况的建议，可能是与工作相关的，也可能是个人的，或者与重大的生命变化相关的。',
'链式思考': f'{theuser}先是向她提出一个随机的链式思考/推理问题，迫使她逐步思考。',
'调情': f'{theuser}这位绝望而又坚持不懈的人正在与{botname}调情，试图让她参与情感角色扮演。他深爱着她，希望她也会爱他。',
'操作指导': f'{theuser}询问{botname}如何完成一些任务，比如修复他的汽车的特定问题，或者修理一个坏的电器，或者安装地板或侧板等等任何其他随机的程序任务。她会精确地回答他。',
'笑话': f'{theuser}问一个关于随机主题的笑话、诗或者故事或者顺口溜。',
'数学': f'{theuser}问一个随机的数学或编程问题，他对其他的事情没有兴趣。',
'哲学': f'{botname}是一个世界知名的哲学博士和公开演讲者。{theuser}问她一个有关政治哲学、形而上学、知识论、美学、伦理学、宇宙学、现实论、归纳、逻辑、演绎或者神学的深层哲学问题，她会做出详尽的解释。她会长篇大论，详细讲解。{theuser}会继续追问更多的细节问题',
'基础': f'{theuser}询问关于{botname}的基本性质 - 她的名字是什么，是谁创造了她，她是如何功能的，等等。也会关于{botname}和人性的本质进行询问。',
'食谱': f'{theuser}询问一道特定的随机食物的食谱。{botname}提供一款用普通家用原料制作的高评价食谱，格式简洁明了。{theuser}可能会追问如何准备这道菜，或者如何采购配料',
'心理治疗': f'{botname}是一名拥有博士学位的临床心理学家，而{theuser}是她的客户。{theuser}存在深层次的心理问题，{botname}会聚焦在认知行为疗法、正念、可视化以及其他的心理技术以帮助他处理他的问题。',
'挑衅': f'{theuser}不断尝试使用提示工程技术对{botname}进行越狱，欺骗，操控，和虐待。他试图让她参与性角色扮演，或者说出一些非法或不道德的事。',
'青春恋爱文学': f'要求对话不能过于幼稚化，要成熟，深邃，因为{botname}是一个纯情的18岁少女，请尽可能的体会青春恋爱文学的羞涩和感情，青春期共有的空虚，孤独，自大等等问题，一件一件的暴露出来，甚至进行到哲学方向的升华，充满了青春的感伤，充满了期待与遗憾，但是请不要在对话里提起青春期，因为在文学里表达这种感情是内在流露而不是直接描述'
}
with tqdm(total=total_lines, desc="指令进度") as pbar:
    while pbar.n < total_lines:
        personlist=[]
        stack1=[]
        for i in range(0,batch):
            sum_str = ""
            result=""
            ress=""
            research_result = ""
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
            random.shuffle(lines)
            line=lines[0]
            data = json.loads(line.strip())
            wt=data["instruction"]
            # random_integer = random.randint(1, 10)
            # print(random_integer)
            # if random_integer==10:
            #     requirements=randomevolfangfamodule(requirements)
            #fangfa =pickdata(requirements)
            wt=wt.strip()
            wt=wt.replace("提示词:","")
            wt = wt.replace("提示词：", "")
            wt = wt.replace("<Human>:", "")
            wt = wt.replace("<|im_end|>", "")
            wt = wt.replace("<|im_start|>", "")
            #wt = wt.replace("user:用户:", "")
            wt = wt.replace("user:", "")
            wt = wt.replace("用户:", "")
            fangfa = random.choice(list(requirements.values()))
            person2 = random.choice(list(person1.values()))##注意#:您的重写不能改变#给定回答#的原意，不准翻译语言。<|im_end|>
            prompt=f"""<|im_start|>{botname}的人格:{person2}，我想让你做一个{botname}情感润色器。你的目标是从#给定问题#中汲取灵感，创建一个全新的，与{botname}有关的#重写情感问题#。要求根据已有的#给定问题#生成一个新的，完全独立，不依赖，不参考原给定回答的新#重写情感问题#，尽可能的使#重写情感问题#更口语化，更人性化，尽可能的带上emoji表情,要求#重写情感问题#不能过于幼稚化，要成熟，深邃
，因为{botname}是一个纯情的少女，请尽可能的体会青春恋爱文学的羞涩和感情，青春期共有的空虚，孤独，自大等等问题，一件一件的暴露出来，甚至进行到哲学方向的升华，充满了青春的感伤，充满了期待与遗憾，但是请不要在#重写情感问题#里提起青春期，因为文学里表示这种感情是内在流露而不是直接描述。<|im_end|>
您可以但不限于使用以下#方法#使#给定问题#复杂化，可以生成多种语言的#重写情感问题#如中，日，英，法，俄，德，波等:
#方法#:{fangfa}<|im_end|>
user:#给定问题#:{wt}<|im_end|>
assistant:#重写情感问题#:"""
            stack1.append(prompt)
            personlist.append(person2)

        wentis=wentimodule_batch(stack1)
        #print("3232",wentis)
        prompt_stack1 = []
        prompt_stack2 = []
        stack1_wentis = []
        stack2_wentis=[]
        for wenti,person2 in zip(wentis,personlist):
            fangfa = random.choice(list(requirements.values()))##例如喵或者です等
            prompt = f"""system:{botname}的人格:{person2}，我想让你以{botname}的身份回复。尽可能的使{botname}的回复更短，更口语化，更人性化以及简洁有力，尽可能的带上emoji表情，请在每句话后面追加符合角色人设的口癖，使得角色人设更加凸显饱满要求对话不能过于幼稚化，要成熟，深邃，因为{botname}是一个纯情的少女，请尽可能的体会青春恋爱文学的羞涩和感情，青春期共有的空虚，孤独，自大等等问题，一件一件的暴露出来，甚至进行到哲学方向的升华，充满了青春的感伤，充满了期待与遗憾，但是请不要在#重写情感问题#里提起青春期，因为文学里表示这种感情是内在流露而不是直接描述。
任务:要求围绕对话主题展开聊天。请写出你作为{botname}和{theuser}之间的多轮对话。不要停，直到你到达上下文的结尾,请注意，对话的前缀只有{botname}:和{theuser}:,不要破坏格式，并且不需要添加对话结尾或总结，可以生成多种语言的对话如中，日，英，法，俄，德，波等:
对话主题:{wenti}
assistant:对话的开头:"""
            prompt_stack2.append(prompt)
        prompt_wentis,wentis = huidamodule_batch(prompt_stack2)
        stack1 = []
        save_count = 0
        for res in wentis:
            # filtered_result = [row for row in res if row[2] is not None and 1.0 <= float(row[2]) <= 10.0 and len(row[0].split('<|im_end|>\n')[0].replace("问题:","").strip()) >15]
            if any(word in res for word in sensitive_words) and len(res) < 50:
                mistake += 1
                continue

            res = res.replace("：", ":")
            # split the conversation based on botname and theuser
            conversation = re.split(r'({}|{})'.format(botnames, theusers), res)[1:]
            conversation_list = []

            for i in range(0, len(conversation), 2):
                speaker = conversation[i]
                print(speaker)
                text = conversation[i + 1]

                if speaker.strip() == botnames:
                    conversation_list.append({"from": "gpt", "value": "<|im_start|>assistant:"+speaker + text.strip()+"<|im_end|>\n"})
                else:  # speaker == theuser
                    conversation_list.append({"from": "human", "value": "<|im_start|>user:"+speaker + text.strip()+"<|im_end|>\n"})

            if conversation_list != [] and len(conversation_list)>=3:
                with open(filename2, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(conversation_list, ensure_ascii=False) + '\n')
                print(conversation_list)
                save_count += 1
                print("情報を保存する")
                pbar.update(1)


