"""Optional road segmentation (Cityscapes SegFormer) for ground-level cameras.
For drone/top-down views rely on the learned flow-field road mask instead."""
import cv2
import numpy as np

_CACHE = {}


def _device(d):
    import torch
    if d is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{d}" if isinstance(d, int) else str(d)


def segformer_road_mask(frame_bgr, model_name, device=None):
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    dev = _device(device)
    if model_name not in _CACHE:
        proc = SegformerImageProcessor.from_pretrained(model_name)
        mdl = SegformerForSemanticSegmentation.from_pretrained(model_name).to(dev).eval()
        _CACHE[model_name] = (proc, mdl)
    proc, mdl = _CACHE[model_name]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    logits = F.interpolate(logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
    return (logits.argmax(1)[0] == 0).cpu().numpy()  # Cityscapes class 0 = road


def tint(frame, mask, color=(255, 140, 0), alpha=0.25):
    out = frame.copy()
    out[mask] = cv2.addWeighted(frame, 1 - alpha, np.full_like(frame, color), alpha, 0)[mask]
    return out
