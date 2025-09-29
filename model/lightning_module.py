# -*- coding: utf-8 -*-
# @Time    :   2025/09/28 11:09:03
# @Author  :   lixumin1030@gmail.com
# @FileName:   lightning_module.py


import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.optim import Adam, AdamW
from transformers import get_cosine_schedule_with_warmup
from dataset import PointReaderData
# from transform import train_transform, test_transform
from PIL import Image


class ModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.max_length = self.config.max_length
        # self.image_root_path = self.config.image_root_path
        self.pretrain_model_path = self.config.pretrain_model_path 
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrain_model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.pretrain_model_path, trust_remote_code=True)

        ##
        # tokens_to_add = {'additional_special_tokens':['<angle>', '</angle>', '<sep>', '<GEO>', '</GEO>', '<val>', '</val>']}
        # self.processor.tokenizer.add_special_tokens(tokens_to_add)
        # self.model.resize_token_embeddings(len(self.processor.tokenizer))


    def training_step(self, batch, batch_idx):
        images= [self.process_img(i, "train") for i in batch[0]]
        questions= batch[1]
        answers = batch[2]
        
        inputs = self.processor(
            text=questions, images=images, return_tensors="pt", padding="max_length", max_length=640, truncation=True
        ) # 图像token就占了 557 
        input_ids, pixel_values = inputs["input_ids"].to(self.device), inputs["pixel_values"].to(self.device)

        labels = self.processor.tokenizer(
            text=answers,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        ).input_ids.to(self.device)
        loss = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values, 
            labels=labels,
            return_dict=True
            ).loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        self.log('loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images= [self.process_img(i, "val") for i in batch[0]]
        questions= batch[1]
        answers = batch[2]

        inputs = self.processor(
            text=questions, images=images, return_tensors="pt", padding=True
        )
        input_ids, pixel_values = inputs["input_ids"].to(self.device), inputs["pixel_values"].to(self.device)

        preds = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values, 
            max_new_tokens=self.max_length,
            num_beams=1,
            )
        preds = self.processor.batch_decode(preds, skip_special_tokens=False)
        scores = []
        for generated_text, answer in zip(preds, answers):
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task="<point>",
                image_size=(
                    pixel_values.shape[-2],
                    pixel_values.shape[-1],
                ),
            )
            print("*"*50)
            print("GT:", answer)
            print("Pred:", parsed_answer["<point>"])
            print("*"*50)

        return scores

    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1.0e-6, weight_decay=0.05)
        # 设置 warmup_steps 和 total_steps
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        # return optimizer

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps, 
            num_training_steps=self.config.num_training_samples_per_epoch*self.config.max_epochs
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',  # 指定更新学习率的间隔是每步还是每个 epoch
            'frequency': 1,
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler_config]
        

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.save_path)
        # self.model.vision_config.model_type = "davit"
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    def process_img(self, image_name, split):
        path = f"{image_name}"
        image = Image.open(path).convert("RGB")
        # if split == "train":
        #     image = Image.fromarray(train_transform(image)) 
        return image


class DataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_data_path = self.config.train_data_path
        self.val_data_path = self.config.val_data_path
        self.train_batch_size = self.config.train_batch_size
        self.val_batch_size = self.config.val_batch_size      
        self.train_dataset = None 
        self.val_dataset = None    
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def setup(self, stage=None):
        self.train_dataset = PointReaderData(
                precomputed_path=self.train_data_path,
            )
        self.val_dataset = PointReaderData(
                precomputed_path=self.val_data_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)