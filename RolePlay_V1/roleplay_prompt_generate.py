from transformers import LlamaTokenizer,AutoModelForCausalLM

import torch
ckpt = 'Baichuan-13B-Chat_4bit'
device = torch.device('cuda')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#tokenizer = LlamaTokenizer.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt,trust_remote_code=True)
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized(ckpt, device_map="auto",trust_remote_code=True).half()
# from transformers.generation.utils import GenerationConfig
# from transformers import BitsAndBytesConfig
# model = AutoModelForCausalLM.from_pretrained(ckpt,
#                                              trust_remote_code=True,
#                                              quantization_config=BitsAndBytesConfig(
#                                                  load_in_4bit=True,
#                                                  bnb_4bit_compute_dtype=torch.bfloat16,
#                                                  bnb_4bit_use_double_quant=True,
#                                                  bnb_4bit_quant_type='nf4'
#                                              ),
#                                              device_map="auto")
with open('filter1.txt', 'r', encoding='utf-8') as f:
    sensitive_words = [line.strip() for line in f.readlines()]
def generate(prompt):
    print("1",prompt,"2")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generate_ids = model.generate(input_ids=input_ids,
    max_length=4096,
    #  do_sample = True,
    # eos_token_id=tokenizer.eos_token_id )
    num_beams=1,
    do_sample=True, top_p=0.9, temperature=0.95, repetition_penalty=1.05, eos_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = output[len(prompt):]
   # print(response)
    print("回答：",response)
    return response


import random
import json
from tqdm import tqdm

filename = "seed_prompt.json"
#filename = "xiaoyu_person_指令2.json"
translations = []
total_lines = 100000
sum_str = ""

with tqdm(total=total_lines, desc="指令进度") as pbar:
    while pbar.n < total_lines:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
        random.shuffle(lines)
        i=0
        sum_str = ""
        for line in lines:
            i+=1
            try:
                data = json.loads(line.strip())
            except:
                print("error:",line.strip())
                continue
            question = data["instruction"]
            sum_str += f"{i}.{question}\n"

            if i == 5: 
                res = generate(f'请续写下面内容，不少于10条。\n{sum_str}')
                res = res.split("\n")
                for result in res:
                    result = result.strip()
                    prefix_length = len(result.split(".")[0]) + 1  # 获取前缀数字的长度，包括后面的点号
                    result = result[prefix_length:]
                    if result == "":
                        continue
                    while any(word in result for word in sensitive_words):
                        res = generate(f'请续写下面内容，不少于10条。\n{sum_str}')
                    json_data = {'instruction': result, "input": "", 'output': ""}
                    # 将数据写入文件
                    with open(filename, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
                        
                    pbar.update(1)
        if pbar.n >= total_lines:
            break

