# character_AI_open
开源版characterai&characterGLM
# roleplay_AI
介绍
基于self-instruct生成的多轮对话roleplay数据，约1k条不同的人格数据和对话

模型基于baichuan13b训练的4bit量化版
https://huggingface.co/Minami-su/roleplay_baichuan-Chat_4bit

存在问题：
1.基于模型自身生成，所以roleplay存在模型本身价值观融入情况，导致roleplay不够真实，不够准确。


https://huggingface.co/datasets/Minami-su/roleplay_multiturn_chat_1k_zh_v0.1

## News

[2023-12-16] 中文数据集 [Anime_novel_datasets](https://huggingface.co/datasets/Minami-su/Anime_novel_datasets) Released! 包含153本动漫小说数据！

[2023-12-04] qwen_7b_roleplay_4bit [Yi_34B_Chat_2bit](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit) Released! You can run it on 11G mem GPU,quantize base QuIP# method, a weights-only quantization method that is able to achieve near fp16 performance using only 2 bits per weight.

[2023-11-30] qwen_7b_roleplay_4bit [qwen_7b_roleplay_4bit](https://huggingface.co/Minami-su/qwen_7b_chat_roleplay_4bit) Released! 
