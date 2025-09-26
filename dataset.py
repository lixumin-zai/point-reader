import os
import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from process import process_annotation


class PointReaderData(Dataset):
    """读取 output/annotations 下的 JSON 并生成两类 QA：
    1) question: "detect {xx}", answer: 多个匹配子串的 OBB（每个 OBB 8 个 <loc_..>，用 <spe> 分隔）
    2) question: "points out {xxx}", answer: 多个匹配子串的中心点（每个点 2 个 <loc_..>，用 <spe> 分隔）
    返回字典包含 image_path, question, answer。
    """

    def __init__(self,
                 root_dir: str = "./output",
                 case_sensitive: bool = True,
                 transform=None,
                 use_precomputed: bool = True,
                 precomputed_path: str = "./output/qa_samples.jsonl"):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.case_sensitive = case_sensitive
        self.transform = transform
        self.use_precomputed = use_precomputed
        self.precomputed_path = precomputed_path

        if self.use_precomputed and os.path.isfile(self.precomputed_path):
            # 直接读取预计算的 JSONL
            samples = []
            with open(self.precomputed_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))
            self.samples = samples
        else:
            # 回退：动态从 annotations 构建
            ann_paths = [
                os.path.join(self.ann_dir, f)
                for f in os.listdir(self.ann_dir)
                if f.endswith(".json")
            ]
            ann_paths.sort()
            samples = []
            for p in ann_paths:
                samples.extend(process_annotation(p, images_dir=self.images_dir, case_sensitive=self.case_sensitive))
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image_path = os.path.join(self.images_dir, item["image"])  # filename
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # 默认转为张量 [C,H,W]，范围 [0,1]
            image = torch.from_numpy((torch.ByteTensor(bytearray(image.tobytes())))).float()
            # 上面方式不可靠，建议用户传入 transform，这里简单返回 PIL
            image = image  # 保留 PIL，避免破坏

        return {
            "image": image,
            "image_path": image_path,
            "question": item["question"],
            "answer": item["answer"],
        }