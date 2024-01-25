from transformers import LlamaTokenizer,AutoModelForCausalLM

import torch
ckpt = 'Baichuan-13B-Chat_4bit'
device = torch.device('cuda')
#tokenizer = LlamaTokenizer.from_pretrained(ckpt)
# from auto_gptq import AutoGPTQForCausalLM
# model = AutoGPTQForCausalLM.from_quantized(ckpt, device_map="auto").half()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    max_length=2048,
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
filename0 = "seed_prompt.json"
filename2 = "roleplay_data.json"
translations = []
total_lines = 10000
sum_str = ""
def getq(sum_str):
    result = generate(sum_str)
    result = result.strip()
    while any(word in result for word in sensitive_words):
        if any(word in result for word in sensitive_words):
            print("error reloop")
        result = generate(sum_str)
        result = result.strip()
    return result
def geta(sum_str):
    result = generate(sum_str)
    result = result.strip()
    while any(word in result for word in sensitive_words):
        if any(word in result for word in sensitive_words):
            print("error reloop")
        result = generate(sum_str)
        result = result.strip()
    return result

max_history_len=1000
with tqdm(total=total_lines, desc="指令进度") as pbar:
    while pbar.n < total_lines:
        count=0
        with open(filename0, "r", encoding="utf-8") as file:
            lines2 = file.readlines()
        random.shuffle(lines2)
        i=0
        count=0
        sum_str = ""
        i=0
        for line2 in lines2:
            i+=1
            data2 = json.loads(line2.strip())
            question3 = data2["instruction"]
            name=question3.split(":")[0]
            name=name.replace("人格","")
            name=name.replace("的","")
            
            history=[]
            for _ in range(6):
                input_text=f'''要求扮演下面角色，并且根据角色的设定内容模仿代入角色相应的对话口吻和风格：{question3}<6>\n'''
                for history_id, history_utr in enumerate(history[-max_history_len:]):
                    input_text = input_text + history_utr + '\n'
                prompt = input_text +f"根据上面内容与{name}发起日常对话，只写出一句即可<6>\n对话:"
                prompt = prompt.strip()
                q=getq(prompt)
                #q=q.replace("人类:","")
                # q=q.replace("答案:","")
                # q=q.replace("说：",":")
                history.append("人类:"+q+"<6>")
                sum_str2=f'''要求扮演下面角色，并且根据角色的设定内容模仿代入角色相应的对话口吻和风格：{question3}<6>\n'''
                for history_id, history_utr in enumerate(history[-max_history_len:]):
                    sum_str2 = sum_str2 + history_utr + '\n'
                sum_str2 = sum_str2+f"{name}:"    
                a=geta(sum_str2)
                history.append(f"{name}:"+a+"<6>")     
                
                
            sum_str2=sum_str2+a
            json_data = {'instruction':sum_str2, "input": "", 'output': ""}
            with open(filename2, 'a', encoding='utf-8') as f:
                f.write('\n')
                f.write(json.dumps(json_data, ensure_ascii=False))
            pbar.update(1)
            if i==6:
                break
        if pbar.n >= total_lines:
            break

