"""Convert UA-DETRAC (XML annotations) -> YOLO format + data/detrac.yaml.

Expected (adjust args if your copy differs):
  --images  DETRAC-train-data/Insight-MVI_20011/img00001.jpg ...
  --ann     DETRAC-Train-Annotations-XML/MVI_20011.xml
Splits by SEQUENCE (not frame) so val frames never leak from train videos.
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--ann", required=True)
ap.add_argument("--out", default="data/detrac_yolo")
ap.add_argument("--step", type=int, default=5, help="keep every Nth frame")
ap.add_argument("--val-frac", type=float, default=0.15)
a = ap.parse_args()

CLS = {"car": 0, "van": 1, "bus": 2, "others": 3}
NAMES = ["car", "van", "bus", "truck"]  # DETRAC 'others' ~ trucks/misc
out, imgs = Path(a.out), Path(a.images)
xmls = sorted(Path(a.ann).glob("*.xml"))
n_val = max(1, int(len(xmls) * a.val_frac))
for idx, xp in enumerate(xmls):
    split = "val" if idx >= len(xmls) - n_val else "train"
    seq = xp.stem
    d = next((p for p in (imgs / seq, imgs / f"Insight-{seq}") if p.exists()), None)
    if d is None:
        print("skip (no images):", seq)
        continue
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    W = H = None
    for fr in ET.parse(xp).getroot().iter("frame"):
        num = int(fr.attrib["num"])
        ip = d / f"img{num:05d}.jpg"
        if num % a.step or not ip.exists():
            continue
        if W is None:
            H, W = cv2.imread(str(ip)).shape[:2]
        lines = []
        for t in fr.iter("target"):
            b, at = t.find("box"), t.find("attribute")
            l, tp, w, h = (float(b.attrib[k]) for k in ("left", "top", "width", "height"))
            c = CLS.get(at.attrib.get("vehicle_type", "others"), 3)
            lines.append(f"{c} {(l + w / 2) / W:.6f} {(tp + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        stem = f"{seq}_{num:05d}"
        shutil.copy(ip, out / "images" / split / f"{stem}.jpg")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
yaml_text = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + \
            "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
Path("data/detrac.yaml").write_text(yaml_text)
print("Wrote data/detrac.yaml")
