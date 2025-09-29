# -*- coding: utf-8 -*-
# @Time    :   2025/09/29 11:49:31
# @Author  :   lixumin1030@gmail.com
# @FileName:   test.py


from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch
import time

model_path = "./ckpt"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cpu")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

prompt = "detect 456"

url = "./sample_00003.png"
image = Image.open(url).convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")

with torch.no_grad():
    st = time.time()
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        do_sample=False,
        num_beams=3
    )
    print(time.time() - st)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(generated_ids.shape)
print(generated_text)
