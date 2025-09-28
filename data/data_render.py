import os
import random
import string
import json
from typing import List, Tuple, Dict, Optional
import math

from PIL import Image, ImageDraw, ImageFont
import tqdm

# 允许的字符集：1-9, a-z, A-Z（不包含0）
ALPHABET = "123456789" + string.ascii_letters
# 常用字体家族白名单（优先选择，避免出现方框/缺字形）
COMMON_FAMILIES = {
    "Arial", "Helvetica", "Verdana", "Georgia", "Times New Roman", "Times",
    "Courier New", "Courier", "Menlo", "Monaco",
    "DejaVu Sans", "DejaVu Serif", "Liberation Sans", "Liberation Serif",
    "Noto Sans", "Noto Serif"
}


def find_system_fonts() -> List[str]:
    """在常见系统字体目录中查找可用字体文件."""
    candidates = []
    search_dirs = [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    exts = {".ttf", ".otf", ".ttc"}
    for d in search_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if os.path.splitext(f)[1].lower() in exts:
                        candidates.append(os.path.join(root, f))
    return candidates


def get_font_family(font_path: str) -> Optional[str]:
    """尝试读取字体的家族名称（失败返回 None）。"""
    try:
        font = ImageFont.truetype(font_path, size=24)
        name = font.getname()[0]  # (family, style)
        return name
    except Exception:
        return None


def prefer_common_fonts(font_paths: List[str], families: Optional[List[str]] = None) -> List[str]:
    """从候选字体中过滤出常用字体家族，若无则返回原列表。
    families: 可选，外部传入优先家族列表，若为 None 则使用 COMMON_FAMILIES。
    """
    selected = []
    base_families = families if families else COMMON_FAMILIES
    common_lower = {f.lower() for f in base_families}
    for p in font_paths:
        fam = get_font_family(p)
        if fam and fam.lower() in common_lower:
            selected.append(p)
    return selected if selected else font_paths


def safe_load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    try:
        if font_path:
            return ImageFont.truetype(font_path, size=size)
    except Exception:
        pass
    # 退回到默认字体（位图字体，尺寸不随size变化）
    return ImageFont.load_default()


def text_size(text: str, font: ImageFont.ImageFont) -> Tuple[int, int, Tuple[int, int, int, int]]:
    # 使用一个临时画布计算文字边界
    tmp_img = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(tmp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w, h, bbox


def rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def random_color(bright: bool = False) -> Tuple[int, int, int]:
    if bright:
        return tuple(random.randint(80, 255) for _ in range(3))
    return tuple(random.randint(0, 200) for _ in range(3))

# 旋转辅助：点绕中心旋转
def rotate_point(px: float, py: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    rad = -math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    x2 = cos_a * dx - sin_a * dy + cx
    y2 = sin_a * dx + cos_a * dy + cy
    return x2, y2

# 旋转后新图的左上角偏移（expand=True时），根据原图四角旋转坐标的最小值计算
def rotation_shift_for_image(w: int, h: int, angle_deg: float) -> Tuple[float, float]:
    cx, cy = w / 2.0, h / 2.0
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    rot = [rotate_point(x, y, cx, cy, angle_deg) for (x, y) in corners]
    min_x = min(p[0] for p in rot)
    min_y = min(p[1] for p in rot)
    return min_x, min_y

# 从未旋转的 axis-aligned bbox 生成倾斜最小矩形的四点（canvas坐标）
def obb_from_abox(abox: Tuple[int, int, int, int], img_w: int, img_h: int, angle_deg: float,
                  x_on_canvas: int, y_on_canvas: int) -> List[List[float]]:
    x1, y1, x2, y2 = abox
    cx, cy = img_w / 2.0, img_h / 2.0
    # 未旋转 bbox 的四角（位于原图坐标系）
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    # 旋转到新坐标
    rot_corners = [rotate_point(x, y, cx, cy, angle_deg) for (x, y) in corners]
    # expand=True 会使新图左上角为 (min_x, min_y)，因此要减去它使坐标转为旋转后图像坐标
    min_x, min_y = rotation_shift_for_image(img_w, img_h, angle_deg)
    shifted = [(x - min_x + x_on_canvas, y - min_y + y_on_canvas) for (x, y) in rot_corners]
    # 返回四点多边形
    return [[float(f"{x:.2f}"), float(f"{y:.2f}")] for (x, y) in shifted]

# 修改：返回 pad，便于计算子字符串的绝对位置
def make_item_image(text: str, font: ImageFont.ImageFont, fill: Tuple[int, int, int]) -> Tuple[Image.Image, Tuple[int, int], int]:
    w, h, bbox = text_size(text, font)
    # 扩展一点边距，避免粘连
    pad = max(1, min(6, max(w, h) // 20))
    W, H = w + 2 * pad, h + 2 * pad
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # 将文本画到可见区域内：(-bbox[0], -bbox[1])能消除字体负偏移
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=fill)
    return img, (W, H), pad


def sample_text(min_len: int, max_len: int) -> str:
    L = random.randint(min_len, max_len)
    return "".join(random.choice(ALPHABET) for _ in range(L))


def place_items_on_canvas(
    canvas_w: int,
    canvas_h: int,
    items: List[Dict],
    max_attempts: int = 800,
) -> List[Dict]:
    placed: List[Dict] = []
    occupied: List[Tuple[int, int, int, int]] = []

    for item in items:
        W, H = item["size"]
        ok = False
        for _ in range(max_attempts):
            x = random.randint(0, max(0, canvas_w - W))
            y = random.randint(0, max(0, canvas_h - H))
            rect = (x, y, x + W, y + H)
            if all(not rects_overlap(rect, r) for r in occupied):
                occupied.append(rect)
                item["x"], item["y"] = x, y
                item["rect"] = rect
                item["center"] = [x + W / 2.0, y + H / 2.0]
                placed.append(item)
                ok = True
                break
        if not ok:
            # 放不下就跳过该item
            continue
    return placed


# 计算单个条目中所有连续子字符串的中心点、轴对齐 bbox 与倾斜最小矩形（画布坐标）
def compute_substring_annotations(item: Dict) -> List[Dict]:
    text: str = item["text"]
    font: ImageFont.ImageFont = item["font"]
    pad: int = item["pad"]
    img: Image.Image = item["img"]  # 原始未旋转小图
    x0: int = item["x"]              # 旋转后小图放置在画布上的左上角
    y0: int = item["y"]
    angle: float = item.get("angle", 0.0)
    W, H = img.size

    # 用临时绘制器获取完整文本 bbox 与前缀宽度
    tmp_img = Image.new("L", (1, 1), 0)
    dtmp = ImageDraw.Draw(tmp_img)
    full_bbox = dtmp.textbbox((0, 0), text, font=font)
    # 绘制起点（已考虑字体负偏移）
    origin_x = pad - full_bbox[0]
    origin_y = pad - full_bbox[1]

    subs: List[Dict] = []
    n = len(text)
    for i in range(n):
        # 计算前缀进位宽度（相对完整文本起点）
        prefix_adv = dtmp.textlength(text[:i], font=font)
        for j in range(i + 1, n + 1):
            s = text[i:j]
            # 为当前子串创建透明覆盖层并绘制，以真实像素获取 bbox（未旋转坐标系）
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.text((origin_x + prefix_adv, origin_y), s, font=font, fill=(255, 255, 255, 255))
            abox = overlay.split()[3].getbbox()
            if not abox:
                continue
            # 倾斜最小矩形：从未旋转 abox 派生 OBB 的四点（画布坐标）
            obb = obb_from_abox(abox, W, H, angle, x_on_canvas=x0, y_on_canvas=y0)
            # 轴对齐 bbox：由 OBB 四点取 min/max 得到（画布坐标）
            xs = [p[0] for p in obb]
            ys = [p[1] for p in obb]
            bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
            # 中心点：四点平均（整体子串中心）
            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0

            # 新增：逐字符中心点（当子串包含多个字符时，计算每个字符的中心）
            char_centers: List[List[float]] = []
            if len(s) >= 1:
                for p in range(len(s)):
                    # 该字符相对当前子串起点的进位宽度
                    char_adv = dtmp.textlength(s[:p], font=font)
                    char_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    cod = ImageDraw.Draw(char_overlay)
                    cod.text((origin_x + prefix_adv + char_adv, origin_y), s[p], font=font, fill=(255, 255, 255, 255))
                    cabox = char_overlay.split()[3].getbbox()
                    if not cabox:
                        continue
                    cobb = obb_from_abox(cabox, W, H, angle, x_on_canvas=x0, y_on_canvas=y0)
                    cxs = [pp[0] for pp in cobb]
                    cys = [pp[1] for pp in cobb]
                    ccx = sum(cxs) / 4.0
                    ccy = sum(cys) / 4.0
                    char_centers.append([float(f"{ccx:.2f}"), float(f"{ccy:.2f}")])

            subs.append({
                "text": s,
                "center": [float(f"{cx:.2f}"), float(f"{cy:.2f}")],
                "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                "obb": obb,
                "char_centers": char_centers,
            })
    return subs


def render_image(
    width: int,
    height: int,
    n_items: int,
    min_len: int,
    max_len: int,
    min_font: int,
    max_font: int,
    fonts: List[str],
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    min_angle: int = -25,
    max_angle: int = 25,
) -> Tuple[Image.Image, List[Dict], List[Dict]]:
    # 先生成所有item图片与尺寸
    raw_items = []
    for _ in range(n_items):
        text = sample_text(min_len, max_len)
        font_path = random.choice(fonts) if fonts else None
        size = random.randint(min_font, max_font)
        font = safe_load_font(font_path, size)
        color = random_color()
        item_img, (W, H), pad = make_item_image(text, font, color)
        # 随机旋转角度
        angle = random.uniform(min_angle, max_angle)
        rotated = item_img.rotate(angle, expand=True, resample=Image.BICUBIC)
        Wr, Hr = rotated.size
        raw_items.append({
            "text": text,
            "img": item_img,        # 原始未旋转小图
            "img_rot": rotated,      # 旋转后小图
            "size": (Wr, Hr),        # 用旋转后尺寸进行布局
            "font": font,
            "pad": pad,
            "angle": angle,
            "orig_size": (W, H),
        })

    placed = place_items_on_canvas(width, height, raw_items)

    # 合成到大图
    canvas = Image.new("RGB", (width, height), bg_color)
    for it in placed:
        # 合成旋转后的小图
        canvas.paste(it["img_rot"], (it["x"], it["y"]), it["img_rot"]) 

    # 输出：条目文本与中心点与bbox/obb
    annotations: List[Dict] = []
    for it in placed:
        # 未旋转小图内的 tight bbox（不含 padding）
        abox0 = it["img"].split()[3].getbbox()
        if abox0 is None:
            abox0 = (0, 0, it["orig_size"][0], it["orig_size"][1])
        # 倾斜最小矩形：四点
        obb = obb_from_abox(abox0, it["orig_size"][0], it["orig_size"][1], it.get("angle", 0.0), it["x"], it["y"]) 
        # 轴对齐 bbox：由 OBB 的四点取 min/max
        xs = [p[0] for p in obb]
        ys = [p[1] for p in obb]
        bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
        # 中心点：四点平均（整体条目中心）
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0

        # 新增：逐字符中心点（针对整条目 text 的每个字符）
        char_centers_item: List[List[float]] = []
        try:
            full_tmp = Image.new("L", (1, 1), 0)
            dfull = ImageDraw.Draw(full_tmp)
            full_bbox = dfull.textbbox((0, 0), it["text"], font=it["font"])  # type: ignore
            origin_x = it["pad"] - full_bbox[0]
            origin_y = it["pad"] - full_bbox[1]
            for pidx in range(len(it["text"])):
                adv = dfull.textlength(it["text"][:pidx], font=it["font"])  # type: ignore
                char_overlay = Image.new("RGBA", it["img"].size, (0, 0, 0, 0))
                cod = ImageDraw.Draw(char_overlay)
                cod.text((origin_x + adv, origin_y), it["text"][pidx], font=it["font"], fill=(255, 255, 255, 255))
                cabox = char_overlay.split()[3].getbbox()
                if not cabox:
                    continue
                cobb = obb_from_abox(cabox, it["orig_size"][0], it["orig_size"][1], it.get("angle", 0.0), it["x"], it["y"])  # type: ignore
                cxs = [pp[0] for pp in cobb]
                cys = [pp[1] for pp in cobb]
                ccx = sum(cxs) / 4.0
                ccy = sum(cys) / 4.0
                char_centers_item.append([float(f"{ccx:.2f}"), float(f"{ccy:.2f}")])
        except Exception:
            pass

        annotations.append({
            "text": it["text"],
            "center": [float(f"{cx:.2f}"), float(f"{cy:.2f}")],
            "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
            "obb": obb,
            "char_centers": char_centers_item,
        })

    # 输出：所有连续子字符串的中心点与bbox/obb
    sub_annotations: List[Dict] = []
    for it in placed:
        sub_annotations.extend(compute_substring_annotations(it))

    return canvas, annotations, sub_annotations


class DatasetRenderer:
    """使用类来控制数据集生成参数，不依赖 argparse。"""
    def __init__(
        self,
        out_dir: str = "./output",
        n_images: int = 100,
        width: int = 640,
        height: int = 480,
        min_items: int = 5,
        max_items: int = 10,
        min_len: int = 1,
        max_len: int = 3,
        min_font: int = 18,
        max_font: int = 48,
        seed: Optional[int] = None,
        fonts_dir: Optional[str] = None,
        prefer_families: Optional[List[str]] = None,
    ):
        self.out_dir = out_dir
        self.n_images = n_images
        self.width = width
        self.height = height
        self.min_items = min_items
        self.max_items = max_items
        self.min_len = min_len
        self.max_len = max_len
        self.min_font = min_font
        self.max_font = max_font
        self.seed = seed
        self.fonts_dir = fonts_dir
        self.prefer_families = prefer_families

    def run(self):
        if self.seed is not None:
            random.seed(self.seed)

        os.makedirs(self.out_dir, exist_ok=True)
        img_dir = os.path.join(self.out_dir, "images")
        ann_dir = os.path.join(self.out_dir, "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        fonts: List[str] = []
        if self.fonts_dir and os.path.isdir(self.fonts_dir):
            fonts = [
                os.path.join(self.fonts_dir, f)
                for f in os.listdir(self.fonts_dir)
                if os.path.splitext(f)[1].lower() in {".ttf", ".otf", ".ttc"}
            ]
        if not fonts:
            fonts = find_system_fonts()

        # 使用类的 prefer_families 进行过滤；若未提供则使用默认 COMMON_FAMILIES
        fonts = prefer_common_fonts(fonts, families=self.prefer_families)

        for i in tqdm.tqdm(range(self.n_images)):
            n_items = random.randint(self.min_items, self.max_items)
            img, anns, sub_anns = render_image(
                self.width,
                self.height,
                n_items,
                self.min_len,
                self.max_len,
                self.min_font,
                self.max_font,
                fonts,
            )
            img_path = os.path.join(img_dir, f"sample_{i:05d}.png")
            ann_path = os.path.join(ann_dir, f"sample_{i:05d}.json")
            img.save(img_path)
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image": os.path.basename(img_path),
                    "items": anns,
                    "substrings": sub_anns,
                }, f, ensure_ascii=False, indent=2)


def main():
    # 通过类来控制参数，不使用 argparse
    renderer = DatasetRenderer(
        out_dir="../output",
        n_images=100,
        width=640,
        height=480,
        min_items=1,
        max_items=10,
        min_len=1,
        max_len=10,
        min_font=18,
        max_font=48,
        seed=1030,
        fonts_dir=None,
        # 你可以传入自己偏好的家族列表，例如：
        # prefer_families=["Arial", "Helvetica", "Menlo"]
        prefer_families=None,
    )
    renderer.run()
    print(f"数据已生成到: {os.path.abspath(renderer.out_dir)}")


if __name__ == "__main__":
    main()