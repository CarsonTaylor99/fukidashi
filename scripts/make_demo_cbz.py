"""Generate a tiny synthetic manga CBZ for testing the pipeline without
real scans. The 5-page story is built to punish context-free translation:
a cat named 大福 (Daifuku — literally "rice cake"), a nickname (ギン for
銀次) that must stay consistent, and speakers with different voices.

Usage: python scripts/make_demo_cbz.py --font /path/to/JP-font.ttf [-o demo.cbz]
"""

import argparse
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 1400

# (bubbles) per page; each bubble: (cx, cy, w, h, text lines)
PAGES = [
    [
        (700, 300, 460, 220, ["大福！", "どこだーっ！"]),
        (300, 800, 420, 180, ["また逃げたのか"]),
        (650, 1150, 460, 180, ["窓が開いてた…"]),
    ],
    [
        (680, 320, 480, 200, ["大福って…", "食べ物？"]),
        (320, 850, 460, 220, ["猫だよ！", "うちの猫！"]),
    ],
    [
        (680, 320, 500, 200, ["ねえ、ギンって", "呼んでいい？"]),
        (320, 800, 400, 160, ["銀次だ"]),
        (640, 1150, 380, 150, ["…まあいい"]),
    ],
    [
        (660, 300, 400, 180, ["いた！大福！"]),
        (330, 900, 420, 180, ["屋根の上か…"]),
    ],
    [
        (660, 320, 520, 240, ["降りてこい大福", "晩飯抜きだぞ"]),
        (320, 900, 480, 200, ["猫に厳しいなあ"]),
    ],
]


def draw_page(bubbles, font):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, W - 30, H - 30], outline="black", width=5)  # panel frame
    d.line([30, H // 2, W - 30, H // 2], fill="black", width=5)      # panel split
    for cx, cy, bw, bh, lines in bubbles:
        d.ellipse([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2],
                  fill="white", outline="black", width=4)
        line_h = font.size + 10
        y = cy - line_h * len(lines) / 2 + 4
        for line in lines:
            tw = d.textlength(line, font=font)
            d.text((cx - tw / 2, y), line, fill="black", font=font)
            y += line_h
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help="path to a Japanese TTF/OTF")
    ap.add_argument("-o", "--out", default="demo.cbz")
    args = ap.parse_args()

    font = ImageFont.truetype(args.font, 40)
    try:
        font.set_variation_by_name("Bold")
    except OSError:
        pass  # not a variable font

    out = Path(args.out)
    with zipfile.ZipFile(out, "w") as zf:
        for i, bubbles in enumerate(PAGES, 1):
            img = draw_page(bubbles, font)
            tmp = out.parent / f"_page{i}.png"
            img.save(tmp)
            zf.write(tmp, f"{i:02d}.png")
            tmp.unlink()
    print(f"wrote {out} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
