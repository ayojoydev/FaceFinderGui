# ============= КОНСТАНТЫ =============
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ============= УТИЛИТЫ =============
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"File is empty: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot decode image: {path} (unsupported format or corrupted)")
    return img

def list_images(root: Path, allowed_exts: set = ALLOWED_EXTS) -> List[Path]:
    allowed = {e.lower() for e in allowed_exts}
    out = []
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for de in it:
                    if de.is_dir(follow_symlinks=False):
                        stack.append(Path(de.path))
                    elif de.is_file(follow_symlinks=False):
                        ext = Path(de.name).suffix.lower()
                        if ext in allowed:
                            out.append(Path(de.path))
        except FileNotFoundError:
            continue
    return sorted(out)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T