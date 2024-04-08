import json


with open('Complex_Evol_Network_Instruct_1.5k_iter1.json', 'r', encoding='utf-8') as file:
    data = file.readlines()

formatted_data = []

for line in data:
    try:
        json_data = json.loads(line)
    except Exception as e:
        print(f"Error: {e}")  # 打印异常信息
        continue
    formatted_data.append({
        "instruction": json_data["instruction"],
        "input": json_data["input"],
        "output": json_data["output"]
    })

with open('less_is_more_cl2_ti2_clformatted.json', 'w', encoding='utf-8') as file:
    json.dump(formatted_data, file, indent=4, ensure_ascii=False)

# 读取json文件
with open('less_is_more_cl2_ti2_clformatted.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 去除重复的指令
instructions = set()
filtered_data = []
for item in data:
    instruction = item['instruction']
    if instruction not in instructions:
        instructions.add(instruction)
        filtered_data.append(item)

# 写入新的json文件
with open('less_is_more_cl2_ti2_clformatted去重.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, indent=4, ensure_ascii=False)



import json

# 打开1.txt文件并读取内容
with open('less_is_more_cl2_ti2_clformatted去重.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将每条数据转换为1.jsonl格式
for item in data:
    context = item['instruction']
    target =  item['output']
    input=''
    jsonl = {"instruction": context,"input":input,"output": target}

    # 将1.jsonl数据写入文件
    with open('Complex_Evol_Network_Instruct_1.5k_iter1去重2.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(jsonl, ensure_ascii=False) + '\n')
