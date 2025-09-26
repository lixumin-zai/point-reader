import os
import random
import string
import json
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageDraw, ImageFont

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


# 新增：计算单个条目中所有连续子字符串的中心点（画布坐标）
def compute_substring_annotations(item: Dict) -> List[Dict]:
     text: str = item["text"]
     font: ImageFont.ImageFont = item["font"]
     pad: int = item["pad"]
     img: Image.Image = item["img"]
     x0: int = item["x"]
     y0: int = item["y"]
 
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
             # 为当前子串创建透明覆盖层并绘制，以真实像素获取 bbox
             overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
             od = ImageDraw.Draw(overlay)
             od.text((origin_x + prefix_adv, origin_y), s, font=font, fill=(255, 255, 255, 255))
             abox = overlay.split()[3].getbbox()  # alpha 通道的非空区域 bbox
             if not abox:
                 continue
             bx1 = x0 + abox[0]
             by1 = y0 + abox[1]
             bx2 = x0 + abox[2]
             by2 = y0 + abox[3]
             cx = (bx1 + bx2) / 2.0
             cy = (by1 + by2) / 2.0
             subs.append({
                 "text": s,
                 "center": [float(f"{cx:.2f}"), float(f"{cy:.2f}")],
                 "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
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
        raw_items.append({
            "text": text,
            "img": item_img,
            "size": (W, H),
            "font": font,
            "pad": pad,
        })

    placed = place_items_on_canvas(width, height, raw_items)

    # 合成到大图
    canvas = Image.new("RGB", (width, height), bg_color)
    for it in placed:
        canvas.alpha_composite(it["img"], (it["x"], it["y"])) if canvas.mode == "RGBA" else canvas.paste(it["img"], (it["x"], it["y"]), it["img"]) 

    # 输出：条目文本与中心点与bbox
    annotations: List[Dict] = []
    for it in placed:
        abox = it["img"].split()[3].getbbox()
        if abox is None:
            bx1, by1, bx2, by2 = it["x"], it["y"], it["x"] + it["size"][0], it["y"] + it["size"][1]
        else:
            bx1, by1, bx2, by2 = it["x"] + abox[0], it["y"] + abox[1], it["x"] + abox[2], it["y"] + abox[3]
        annotations.append({
            "text": it["text"],
            "center": [float(f"{it['center'][0]:.2f}"), float(f"{it['center'][1]:.2f}")],
            "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
        })

    # 输出：所有连续子字符串的中心点与bbox
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

        for i in range(self.n_images):
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
        out_dir="./output",
        n_images=100,
        width=640,
        height=480,
        min_items=5,
        max_items=10,
        min_len=1,
        max_len=10,
        min_font=18,
        max_font=48,
        seed=None,
        fonts_dir=None,
        # 你可以传入自己偏好的家族列表，例如：
        # prefer_families=["Arial", "Helvetica", "Menlo"]
        prefer_families=None,
    )
    renderer.run()
    print(f"数据已生成到: {os.path.abspath(renderer.out_dir)}")


if __name__ == "__main__":
    main()