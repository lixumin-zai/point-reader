import os
import json
from typing import List, Dict, Any

from torch.utils.data import Dataset


class PointReaderData(Dataset):
    """读取 output/annotations 下的 JSON 并生成两类 QA：
    1) question: "detect {xx}", answer: 多个匹配子串的 OBB（每个 OBB 8 个 <loc_..>，用 <spe> 分隔）
    2) question: "points out {xxx}", answer: 多个匹配子串的中心点（每个点 2 个 <loc_..>，用 <spe> 分隔）
    返回字典包含 image_path, question, answer。
    """

    def __init__(self,images_dir="./output/images",
                 precomputed_path: str = "./output/qa_samples.jsonl"):
        self.images_dir = images_dir
        self.precomputed_path = precomputed_path

        # 直接读取预计算的 JSONL
        samples = []
        with open(self.precomputed_path, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        image = os.path.join(self.images_dir, item["image"])
        question = item["question"]
        answer = item["answer"]

        return image, question, answer