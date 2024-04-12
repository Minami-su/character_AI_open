"""
You need use this peft branch:https://github.com/fahadh4ilyas/peft/tree/hqq-lora
command
python ft-bit--hqq-lora-sft.py \
    --base_model 'Qwen1.5-32B-Chat_llamafy' \
    --data_path 'alpaca_dpo_1k.json' \
    --output_dir 'alpaca_dpo_1k' \
    --batch_size 1 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 5e-6 \
    --cutoff_len 768 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
    
"""
import os
import sys
import types
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import os

from peft import LoraConfig, get_peft_model

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from hqq.core.peft import PeftUtils
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from utils.prompter import Prompter
import signal
import sys
import os

from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora import (
	LoraConfig,
	LoraModel,
	QuantLinear as PeftQuantLinear
)
def find_all_linear_names(model):
	lora_module_names = set()
	for name, module in model.named_modules():
		#print(name, module)
		if isinstance(module, HQQLinear):
			names = name.split('.')
			lora_module_names.add(names[-1])
	return list(lora_module_names)
os.environ["WANDB_DISABLED"] = "true"


def train(
		# model/data params
		base_model: str = "",  # the only required argument
		data_path: str = "yahma/alpaca-cleaned",
		output_dir: str = "./lora-alpaca",
		# training hyperparams
		batch_size: int = 128,
		micro_batch_size: int = 4,
		num_epochs: int = 3,
		learning_rate: float = 3e-4,
		cutoff_len: int = 256,
		val_set_size: int = 2000,
		# lora hyperparams
		lora_r: int = 8,
		lora_alpha: int = 16,
		lora_dropout: float = 0.05,
		log_steps: int = 10,
		# llm hyperparams
		train_on_inputs: bool = True,  # if False, masks out inputs in loss
		add_eos_token: bool = False,
		group_by_length: bool = False,  # faster, but produces an odd training loss curve
		# wandb params
		wandb_project: str = "",
		wandb_run_name: str = "",
		wandb_watch: str = "",  # options: false | gradients | all
		wandb_log_model: str = "",  # options: false | true
		resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
		prompt_template_name: str = "alpaca2",  # The prompt template to use, will default to alpaca.
):
	if int(os.environ.get("LOCAL_RANK", 0)) == 0:
		print(
			f"Training Alpaca-LoRA model with params:\n"
			f"base_model: {base_model}\n"
			f"data_path: {data_path}\n"
			f"output_dir: {output_dir}\n"
			f"batch_size: {batch_size}\n"
			f"micro_batch_size: {micro_batch_size}\n"
			f"num_epochs: {num_epochs}\n"
			f"learning_rate: {learning_rate}\n"
			f"cutoff_len: {cutoff_len}\n"
			f"val_set_size: {val_set_size}\n"
			f"lora_r: {lora_r}\n"
			f"lora_alpha: {lora_alpha}\n"
			f"lora_dropout: {lora_dropout}\n"
			f"train_on_inputs: {train_on_inputs}\n"
			f"add_eos_token: {add_eos_token}\n"
			f"group_by_length: {group_by_length}\n"
			f"wandb_project: {wandb_project}\n"
			f"wandb_run_name: {wandb_run_name}\n"
			f"wandb_watch: {wandb_watch}\n"
			f"wandb_log_model: {wandb_log_model}\n"
			f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
			f"prompt template: {prompt_template_name}\n"
		)
	assert (
		base_model
	), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
	gradient_accumulation_steps = batch_size // micro_batch_size

	prompter = Prompter(prompt_template_name)

	# Check if parameter passed or if set within environ
	use_wandb = len(wandb_project) > 0 or (
			"WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
	)
	# Only overwrite environ if wandb param passed
	if len(wandb_project) > 0:
		os.environ["WANDB_PROJECT"] = wandb_project
	if len(wandb_watch) > 0:
		os.environ["WANDB_WATCH"] = wandb_watch
	if len(wandb_log_model) > 0:
		os.environ["WANDB_LOG_MODEL"] = wandb_log_model
	use_wandb=None
	tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

	if base_model.find("qwen") != -1 or base_model.find("Qwen") != -1:
		tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
		tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
		tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

	else:
		tokenizer.pad_token_id = (
			0  # unk. we want this to be different from the eos token
		)
		tokenizer.bos_token_id = (
			1  # unk. we want this to be different from the eos token
		)
		tokenizer.eos_token_id = (
			2  # unk. we want this to be different from the eos token
		)
	tokenizer.padding_side = "left"  # Allow batched inference

	def save_model(signal, frame):
		print("\nSaving the model...")
		model.save_pretrained(output_dir)
		sys.exit(0)

	def tokenize(prompt, add_eos_token=True):
		# there's probably a way to do this with the tokenizer settings
		# but again, gotta move fast
		result = tokenizer(
			prompt,
			truncation=True,
			max_length=cutoff_len,
			padding=False,
			return_tensors=None,
		)
		if (
				result["input_ids"][-1] != tokenizer.eos_token_id
				and len(result["input_ids"]) < cutoff_len
				and add_eos_token
		):
			result["input_ids"].append(tokenizer.eos_token_id)
			result["attention_mask"].append(1)

		result["labels"] = result["input_ids"].copy()

		return result

	def generate_and_tokenize_prompt(data_point):
		full_prompt = prompter.generate_prompt(
			data_point["instruction"],
			data_point["input"],
			data_point["output"],
		)
		tokenized_full_prompt = tokenize(full_prompt)

		return tokenized_full_prompt

	print(tokenizer.pad_token_id)
	print(tokenizer.pad_token)
	print(tokenizer.bos_token_id)
	print(tokenizer.bos_token)
	print(tokenizer.eos_token_id)
	print(tokenizer.eos_token)
	if data_path.endswith(".json") or data_path.endswith(".jsonl"):
		data = load_dataset("json", data_files=data_path)
	else:
		data = load_dataset(data_path)
	if val_set_size > 0:
		train_val = data["train"].train_test_split(
			test_size=val_set_size, shuffle=True, seed=42
		)
		train_data = (
			train_val["train"].shuffle().map(generate_and_tokenize_prompt)
		)
		val_data = (
			train_val["test"].shuffle().map(generate_and_tokenize_prompt)
		)
	else:
		train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
		val_data = None


	# HQQ Quantize
	######################################################################################
# 	model = HQQModelForCausalLM.from_pretrained(base_model)
# 	tokenizer = AutoTokenizer.from_pretrained(base_model)
# 	# Quantize the model

# 	quant_config = BaseQuantizeConfig(nbits=1, group_size=8, quant_scale=False, quant_zero=False)
# 	model.quantize_model(quant_config=quant_config)
	#model = AutoHQQHFModel.from_quantized(base_model).half().cuda()
	device = 'cuda' # your cude device
	compute_dtype = torch.float16 # dtype: float16, bfloat16
	model = AutoHQQHFModel.from_quantized(base_model, device=device, compute_dtype=compute_dtype)
	# Config Lora
	# transformers trainer will try to read hf_quantizer.is_trainable
	# so we hack it by adding a fake hf_quantizer
	model.is_quantized = True
	#model._is_quantized_training_enabled = True
	model.hf_quantizer = types.SimpleNamespace(is_trainable=True)
	modules = find_all_linear_names(model)
	print(modules)
	# Add Peft
	######################################################################################
	config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=modules,
		lora_dropout=lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, config)

	#model=enable_gradients(model)
	# HQQLinear.set_backend(HQQBackend.PYTORCH)          #Pytorch backend
	# HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)  #Compiled Pytorch via dynamo
	# HQQLinear.set_backend(HQQBackend.ATEN)
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
	trainer = transformers.Trainer(
		model= model,
		train_dataset=train_data,
		eval_dataset=val_data,
		args=transformers.TrainingArguments(
			per_device_train_batch_size=micro_batch_size,
			gradient_accumulation_steps=gradient_accumulation_steps,
			warmup_steps=0,
			num_train_epochs=num_epochs,
			learning_rate=learning_rate,
			fp16=True,
			logging_steps=log_steps,
			optim='paged_adamw_8bit',
            #optim_target_modules=["attn", "mlp"],#['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit', 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit', 'galore_adamw', 'galore_adamw_8bit', 'galore_adafactor', 'galore_adamw_layerwise', 'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise']
			#gradient_checkpointing=True,
			#gradient_checkpointing_kwargs={'use_reentrant': True},
			evaluation_strategy="steps" if val_set_size > 0 else "no",
			save_strategy="steps",
			eval_steps=100 if val_set_size > 0 else None,
			save_steps=200,
			output_dir=output_dir,
			save_total_limit=2,
			load_best_model_at_end=True if val_set_size > 0 else False,
			group_by_length=group_by_length,
			report_to="wandb" if use_wandb else None,
			run_name=wandb_run_name if use_wandb else None,
			max_grad_norm=1.0,
		),
        #data_collator=data_collator
		data_collator=transformers.DataCollatorForSeq2Seq(
		tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
		),
	)



	signal.signal(signal.SIGINT, save_model)
	model.train()
	trainer.train()
	model.save_pretrained(output_dir)

	print(
		"\n If there's a warning about missing keys above, please disregard :)"
	)


if __name__ == "__main__":
	fire.Fire(train)
