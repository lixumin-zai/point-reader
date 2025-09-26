
import os
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image
import random

BINS_W = 1000
BINS_H = 1000


def _quantize_xy(x: float, y: float, width: int, height: int,
                 bins_w: int = BINS_W, bins_h: int = BINS_H) -> Tuple[int, int]:
    qx = int(max(0, min(bins_w - 1, (x / width) * bins_w)))
    qy = int(max(0, min(bins_h - 1, (y / height) * bins_h)))
    return qx, qy



def _format_point_token(x: float, y: float, width: int, height: int) -> str:
    qx, qy = _quantize_xy(x, y, width, height)
    return f"<loc_{qx}><loc_{qy}>"



def _format_obb_tokens(obb: List[List[float]], width: int, height: int) -> str:
    # obb 为四点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，按给定顺序展开为 8 个 <loc_..>
    tokens: List[str] = []
    for (x, y) in obb:
        tokens.append(_format_point_token(x, y, width, height))
    return "".join(tokens)

# ---------- geometry helpers for "what text on point" ----------


def _point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # 检查边与水平射线相交（ray casting）
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        if cond:
            inside = not inside
    return inside



def _sample_point_in_triangle(a: Tuple[float, float],
                              b: Tuple[float, float],
                              c: Tuple[float, float]) -> Tuple[float, float]:
    r1 = random.random()
    r2 = random.random()
    sqrt_r1 = r1 ** 0.5
    x = (1 - sqrt_r1) * a[0] + sqrt_r1 * (1 - r2) * b[0] + sqrt_r1 * r2 * c[0]
    y = (1 - sqrt_r1) * a[1] + sqrt_r1 * (1 - r2) * b[1] + sqrt_r1 * r2 * c[1]
    return x, y



def _sample_point_in_quad(quad: List[List[float]]) -> Tuple[float, float]:
    # 三角化为 (p0,p1,p2) 或 (p0,p2,p3)，按面积比例随机选取
    p0, p1, p2, p3 = quad
    # 计算两个三角形的近似面积（使用叉乘的绝对值一半）
    def tri_area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
    area1 = tri_area(p0, p1, p2)
    area2 = tri_area(p0, p2, p3)
    total = area1 + area2
    if total <= 1e-9:
        # 面积太小，退化为随机选择顶点附近
        return p0[0], p0[1]
    if random.random() < (area1 / total):
        return _sample_point_in_triangle(tuple(p0), tuple(p1), tuple(p2))
    else:
        return _sample_point_in_triangle(tuple(p0), tuple(p2), tuple(p3))



def _sample_point_in_rect(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    rx = random.uniform(min(x1, x2), max(x1, x2))
    ry = random.uniform(min(y1, y2), max(y1, y2))
    return rx, ry



def _random_point_outside(obbs: List[List[List[float]]], width: int, height: int, max_try: int = 200) -> Optional[Tuple[float, float]]:
    for _ in range(max_try):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        hit = False
        for poly in obbs:
            if _point_in_polygon(x, y, poly):
                hit = True
                break
        if not hit:
            return x, y
    return None



def process_annotation(path: str,
                       images_dir: str = "./output/images",
                       case_sensitive: bool = True,
                       max_questions: Optional[int] = None) -> List[Dict]:
    """
    将生成的标注（output/annotations/*.json）解析为 Q/A 样本：
    - question: "detect {xx}"，answer: 多个符合字符的 obb，用 <spe> 分隔，每个 obb 由 8 个 <loc_..> 组成
    - question: "points out {xxx}"，answer: 多个符合字符的中心点，用 <spe> 分隔，每个点由 2 个 <loc_..> 组成
    - question: "what text on point <loc_..><loc_..>"，answer: 该点所在单个字符（正样本）；或 "None"（负样本）
    返回同一张图的多个样本。
    如果提供 max_questions，则对该图片的样本随机下采样到最多 max_questions 条。
    """
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)

    image_name = info.get("image")
    image_path = os.path.join(images_dir, image_name)
    img = Image.open(image_path)
    width, height = img.size

    substrings: List[Dict] = info.get("substrings", [])

    # 根据是否区分大小写构建分组键
    def norm_text(t: str) -> str:
        return t if case_sensitive else t.lower()

    grouped: Dict[str, List[Dict]] = {}
    original_text_map: Dict[str, str] = {}
    for s in substrings:
        t = s.get("text", "")
        key = norm_text(t)
        grouped.setdefault(key, []).append(s)
        # 记录一个原始文本用于 question 显示
        if key not in original_text_map:
            original_text_map[key] = t

    samples: List[Dict] = []
    for key, entries in grouped.items():
        display_text = original_text_map.get(key, key)

        # detect 问题：输出多个 OBB
        obb_segments: List[str] = []
        for s in entries:
            obb = s.get("obb")
            if not obb:
                # 回退到 bbox 的四角
                x1, y1, x2, y2 = s.get("bbox", [0, 0, 0, 0])
                obb = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            obb_segments.append(_format_obb_tokens(obb, width, height))
        detect_answer = "<spe>".join(obb_segments) if obb_segments else "None"
        samples.append({
            "image": image_name,
            "question": f"detect {display_text}",
            "answer": detect_answer,
        })

        # points 问题：输出多个中心点
        point_segments: List[str] = []
        for s in entries:
            cx, cy = s.get("center", [None, None])
            if cx is None or cy is None:
                # 回退到 bbox 中心
                x1, y1, x2, y2 = s.get("bbox", [0, 0, 0, 0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
            point_segments.append(_format_point_token(cx, cy, width, height))
        points_answer = "<spe>".join(point_segments) if point_segments else "None"
        samples.append({
            "image": image_name,
            "question": f"points out {display_text}",
            "answer": points_answer,
        })

    # 预先收集所有字符（包含多字符）的 OBB，用于负样本点外采样
    all_obbs: List[List[List[float]]] = []
    for s in substrings:
        obb = s.get("obb")
        if obb and len(obb) == 4:
            all_obbs.append(obb)
        else:
            x1, y1, x2, y2 = s.get("bbox", [0, 0, 0, 0])
            all_obbs.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # what text on point：正样本（点在单个字符 OBB 内，点为随机值、非中心）
    positive_count = 0
    for s in substrings:
        t = s.get("text", "")
        if len(t) != 1:
            # 仅针对单字符构造正样本，确保答案为单个字符
            continue
        obb = s.get("obb")
        if obb and len(obb) == 4:
            px, py = _sample_point_in_quad(obb)
        else:
            x1, y1, x2, y2 = s.get("bbox", [0, 0, 0, 0])
            px, py = _sample_point_in_rect(x1, y1, x2, y2)
        token = _format_point_token(px, py, width, height)
        samples.append({
            "image": image_name,
            "question": f"what text on point {token}",
            "answer": t,
        })
        positive_count += 1

    # what text on point：负样本（点不在任何字符 OBB 内，答案为 None）
    # 负样本数量与正样本数量对齐，最多等同于正样本数
    for _ in range(positive_count):
        pt = _random_point_outside(all_obbs, width, height, max_try=300)
        if pt is None:
            continue
        nx, ny = pt
        token = _format_point_token(nx, ny, width, height)
        samples.append({
            "image": image_name,
            "question": f"what text on point {token}",
            "answer": "None",
        })

    # 随机抽取至最多 max_questions 条（仅对当前图片生效）
    if max_questions is not None and len(samples) > max_questions:
        samples = random.sample(samples, k=max_questions)

    return samples



def build_precomputed_dataset(
    out_path: str = "./output/qa_samples.jsonl",
    ann_dir: str = "./output/annotations",
    images_dir: str = "./output/images",
    case_sensitive: bool = True,
    max_questions_per_image: Optional[int] = 10,
) -> int:
    """将所有标注转换为预计算的 JSONL，每行一个样本：
    {"image": str, "question": str, "answer": str}
    可设置 max_questions_per_image，随机选取每张图片最多该数量的问题。
    返回写入的样本数量。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ann_paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith(".json")]
    ann_paths.sort()

    count = 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for p in ann_paths:
            samples = process_annotation(
                p,
                images_dir=images_dir,
                case_sensitive=case_sensitive,
                max_questions=max_questions_per_image,
            )
            for s in samples:
                wf.write(json.dumps(s, ensure_ascii=False) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    # 预计算并写入 JSONL，供 Dataset 直接读取
    out_jsonl = "../output/qa_samples.jsonl"
    n = build_precomputed_dataset(out_path=out_jsonl, max_questions_per_image=10)
    print(f"Precomputed samples: {n}, saved to: {os.path.abspath(out_jsonl)}")