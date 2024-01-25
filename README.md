# character_AI_open
开源版characterai&characterGLM

# roleplay_AI 介绍
基于self-instruct生成的多轮对话roleplay数据，约1k条不同的人格数据和对话

## Getting Started
1.首先生产roleplay的prompt人设设定，这里我上传了seed_prompt.json然后运行代码即可继续生产人设prompt,seed_prompt.json的指令你也可以自己写大概10条就够启动了
```bash
python roleplay_prompt_generate.py
```
2.然后生产多轮对话，这时候运行代码即可生产最终数据
```bash
python roleplay_Multi-round_dialog_generation2.py
```

## 存在问题：
1.基于模型自身生成，所以roleplay存在模型本身价值观融入情况，导致roleplay不够真实，不够准确。并且对模型较为熟悉的人设模仿效果会更好，例如贝多芬，莫扎特等名人，而模型不是很熟悉的人物则生产的数据以及训练后的模仿效果较差。这里的roleplay数据的本质思想是让大模型学会适应roleplay

## 已上传的模型
模型基于baichuan13b训练的4bit量化版
https://huggingface.co/Minami-su/roleplay_baichuan-Chat_4bit

## 1k数据
https://huggingface.co/datasets/Minami-su/roleplay_multiturn_chat_1k_zh_v0.1


# character_AI_open
Open source version of characterai&characterGLM

# roleplay_AI Introduction
Based on self-instructed generated multi-turn dialogue roleplay data, approximately 1k different personality data and conversations.

## Getting Started
1. First, generate the roleplay prompt character settings. I have uploaded seed_prompt.json here, run the code to continue generating character prompts.You can also write approximately 10 instructions for seed_prompt.json yourself, and that should be enough to get started.
```bash
python roleplay_prompt_generate.py
```
2. Then, generate multi-turn dialogues. Run the code at this point to produce the final data.
```bash
python roleplay_Multi-round_dialog_generation2.py
```

## Issues:
1. Due to being based on model-generated content, roleplay may incorporate the model's own values, making it less realistic and accurate. The imitation effect is better for personalities the model is more familiar with, such as famous figures like Beethoven and Mozart. Characters less familiar to the model result in poorer data generation and imitation after training. The fundamental idea behind roleplay data is to enable the large model to adapt to roleplay scenarios.

## Uploaded Models
The model is based on a 4-bit quantized version trained on baichuan13b.
[Roleplay Model - Hugging Face](https://huggingface.co/Minami-su/roleplay_baichuan-Chat_4bit)

## 1k Data
[Roleplay Multiturn Chat 1k Data - Hugging Face](https://huggingface.co/datasets/Minami-su/roleplay_multiturn_chat_1k_zh_v0.1)

Please note that the above content has been revised to English while maintaining the original format.

## News

[2023-12-16] 中文数据集 [Anime_novel_datasets](https://huggingface.co/datasets/Minami-su/Anime_novel_datasets) Released! 包含153本动漫小说数据！

[2023-12-04] qwen_7b_roleplay_4bit [Yi_34B_Chat_2bit](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit) Released! You can run it on 11G mem GPU,quantize base QuIP# method, a weights-only quantization method that is able to achieve near fp16 performance using only 2 bits per weight.

[2023-11-30] qwen_7b_roleplay_4bit [qwen_7b_roleplay_4bit](https://huggingface.co/Minami-su/qwen_7b_chat_roleplay_4bit) Released! 
