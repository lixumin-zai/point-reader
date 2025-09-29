# -*- coding: utf-8 -*-
# @Time    :   2025/09/29 11:49:31
# @Author  :   lixumin1030@gmail.com
# @FileName:   test.py


import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch
import time

logger = logging.getLogger(__name__)

class PointReader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def __call__(self, prompt, image):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            st = time.time()
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                do_sample=False,
                num_beams=3
            )
            logger.info(f"生成时间: {time.time() - st}")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            logger.info(f"生成文本: {generated_text}")
        return generated_text


