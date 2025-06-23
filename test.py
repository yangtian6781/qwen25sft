import torch
from transformers import Qwen2_5_VLProcessor
qwen25_processor = Qwen2_5_VLProcessor.from_pretrained("/share/tianyang/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)
a = torch.load("/share/tianyang/qwen25vl/zhushidaima.pt", weights_only=False)
b = torch.load("/share/tianyang/qwen25vl/quanbudaima.pt", weights_only=False)
v = 12
