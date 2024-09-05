# -*- coding: utf-8 -*-
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vita.constants import GLOBAL_WEIGHTS_PATH

model_dir = f"{GLOBAL_WEIGHTS_PATH}/Mixtral-8x7B_modVocab/mg2hg"
tokenizer = AutoTokenizer.from_pretrained(model_dir)


system_prompt = "你是一个人工智能机器人。\n- 你是研究社区开发的大语言模型。你的设计宗旨是有益、诚实且无害。\n- 你支持使用用户选择的多种语言流利地进行交流并解答用户的问题。\n- 如果用户更正你生成的错误答案，你会向用户致歉并与用户探讨正确的答案。"

question = "请详细介绍一下火星。"

chat_template = "system:{system_prompt}</s>\nuser:{question}</s>\nbot:"

text = chat_template.format(system_prompt=system_prompt, question=question)
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
input_ids = input_ids.to("cuda")


model = AutoModelForCausalLM.from_pretrained(
    #    model_dir, torch_dtype=torch.float16, device_map="auto",attn_implementation="flash_attention_2").eval()
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

start_time = time.time()
outputs = model.generate(input_ids, max_new_tokens=10)
time_consume = time.time() - start_time

outputs = outputs.cpu().numpy()[0]
outputs = outputs[len(input_ids[0]) :]
output_text = tokenizer.decode(outputs, skip_special_tokens=True)


print(output_text)
print(f"Time consume: {time_consume}")
