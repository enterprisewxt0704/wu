"""统计训练/验证数据并生成8张样本预览图。"""
import argparse, csv, random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()
with (ROOT / "labels_train_val.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
splits = Counter(r["split"] for r in rows)
classes = Counter(r["class_cn"] for r in rows)
print(f"train={splits['train']} val={splits['val']}")
print("类别统计：" + "，".join(f"{k}={v}" for k, v in sorted(classes.items())))
canvas = Image.new("RGB", (1024, 572), "white")
draw = ImageDraw.Draw(canvas)
for i, row in enumerate(random.Random(20260901).sample(rows, 8)):
    im = Image.open(ROOT / row["relative_path"]).convert("RGB").resize((256, 256))
    x, y = (i % 4) * 256, (i // 4) * 286
    canvas.paste(im, (x, y + 28))
    draw.text((x + 8, y + 7), f"{row['class_name']} / {row['severity']}", fill=(20, 40, 60))
preview = ROOT / "样本预览.png"
canvas.save(preview)
print(f"预览图：{preview}")
if not args.no_show:
    canvas.show()

