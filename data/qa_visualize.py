import os
import re
import json
import argparse
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw

BINS_W = 1000
BINS_H = 1000

LOC_RE = re.compile(r"<loc_(\d+)>")


def decode_points_from_tokens(token_str: str, width: int, height: int) -> List[Tuple[float, float]]:
    """
    将像素量化 token 序列解码为像素坐标点列表。
    token_str: 包含若干 <loc_x><loc_y> 的字符串
    返回: [(x, y), ...]
    """
    ids = [int(m.group(1)) for m in LOC_RE.finditer(token_str)]
    pts: List[Tuple[float, float]] = []
    for i in range(0, len(ids) - 1, 2):
        qx, qy = ids[i], ids[i + 1]
        # 对应编码时 qx in [0..BINS_W-1]，反量化到像素，使用bin中心
        x = (qx + 0.5) / BINS_W * width
        y = (qy + 0.5) / BINS_H * height
        pts.append((x, y))
    return pts


def draw_point(draw: ImageDraw.ImageDraw, p: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0), r: int = 4):
    x, y = p
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def draw_polygon(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], color: Tuple[int, int, int] = (0, 255, 0)):
    if len(pts) >= 2:
        draw.line(pts + [pts[0]], fill=color, width=2)


def visualize_sample(idx: int, sample: dict, images_dir: str, out_dir: str):
    image_name = sample.get("image")
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        return
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    q = sample.get("question", "")
    a = sample.get("answer", "")

    # detect
    if q.startswith("detect "):
        # answer: 多个 OBB，用 <spe> 分隔，每个 obb 由 8 个 <loc_..>
        if a == "None":
            pass
        else:
            segments = a.split("<spe>") if "<spe>" in a else [a]
            colors = [(0, 255, 0), (255, 165, 0), (135, 206, 235), (255, 105, 180)]
            for si, seg in enumerate(segments):
                pts = decode_points_from_tokens(seg, w, h)
                # 取前4个点画多边形
                poly = pts[:4]
                color = colors[si % len(colors)]
                draw_polygon(draw, poly, color=color)
    # points out
    elif q.startswith("points out "):
        # answer: 多个点，用 <spe> 分隔，每个点 2 个 <loc_..>
        if a != "None":
            segments = a.split("<spe>") if "<spe>" in a else [a]
            colors = [(255, 0, 0), (0, 191, 255), (255, 215, 0), (124, 252, 0)]
            for si, seg in enumerate(segments):
                pts = decode_points_from_tokens(seg, w, h)
                for p in pts:
                    draw_point(draw, p, color=colors[si % len(colors)])
    # what text on point
    elif q.startswith("what text on point "):
        # question 中包含一个点
        token_part = q[len("what text on point "):]
        pts = decode_points_from_tokens(token_part, w, h)
        if pts:
            draw_point(draw, pts[0], color=(255, 255, 0))
        # 若答案为字符，则在点附近画一个小标注；若为 None，写 None
        ans = a
        label = ans if ans != "None" else "None"
        if pts:
            x, y = pts[0]
            draw.text((x + 6, y - 12), str(label), fill=(255, 255, 0))

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    safe_qtype = (
        "detect" if q.startswith("detect ") else
        "points" if q.startswith("points out ") else
        "pointtext" if q.startswith("what text on point ") else
        "other"
    )
    base = os.path.splitext(os.path.basename(image_name))[0]
    out_name = f"{idx:06d}_{base}_{safe_qtype}.png"
    img.save(os.path.join(out_dir, out_name))


def main():
    parser = argparse.ArgumentParser(description="Visualize QA samples from JSONL")
    parser.add_argument("--jsonl", default="../output/qa_samples.jsonl", help="Path to qa_samples.jsonl")
    parser.add_argument("--images", default="../output/images", help="Images directory")
    parser.add_argument("--out", default="../output/qa_viz", help="Output directory for visualizations")
    parser.add_argument("--max", type=int, default=100, help="Max number of samples to render")
    parser.add_argument("--filter", type=str, default="all", choices=["all", "detect", "points", "pointtext"], help="Filter by question type")
    args = parser.parse_args()

    count = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue
            q = sample.get("question", "")
            qtype = (
                "detect" if q.startswith("detect ") else
                "points" if q.startswith("points out ") else
                "pointtext" if q.startswith("what text on point ") else
                "other"
            )
            if args.filter != "all" and qtype != args.filter:
                continue
            visualize_sample(idx, sample, args.images, args.out)
            count += 1
            if count >= args.max:
                break

    print(f"Saved {count} visualizations to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()