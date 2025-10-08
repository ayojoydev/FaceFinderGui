from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2


from sklearn.cluster import AgglomerativeClustering
from insightface.app import FaceAnalysis


from utils import (
    imread_bgr, list_images, l2_normalize, cosine_sim, ALLOWED_EXTS
)


def extract_faces(app: FaceAnalysis, img_bgr: np.ndarray):
    faces = app.get(img_bgr)
    out = []
    for f in faces:
        emb = f.normed_embedding if hasattr(f, "normed_embedding") else f.embedding
        out.append({
            "bbox": f.bbox.astype(int).tolist(),
            "kps": f.kps.tolist(),
            "det_score": float(f.det_score),
            "embedding": emb.astype(np.float32)
        })
    return out

def init_face(det_size=(640, 640)) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=det_size)
    return app

def build_catalog_from_sets(app: FaceAnalysis, sets_root: Path, min_det_score=0.35,
                            cluster_dist=0.55, progress_callback=None):
    set_dirs = [p for p in sets_root.iterdir() if p.is_dir()]
    if not set_dirs:
        raise RuntimeError(f"No subfolders found in {sets_root}")

    set_names = [sd.name for sd in sorted(set_dirs)]
    all_image_paths = []
    for sd in sorted(set_dirs):
        all_image_paths.extend(list_images(sd))

    if not all_image_paths:
        raise RuntimeError("No images found in sets")

    if progress_callback:
        progress_callback(f"Найдено изображений: {len(all_image_paths)}. Обработка...")

    # === Последовательная обработка ВСЕХ изображений ===
    all_embs = []
    all_meta = []

    for i, path in enumerate(all_image_paths):
        try:
            if progress_callback and i % 20 == 0:
                progress_callback(f"Обработано: {i}/{len(all_image_paths)}")

            img = imread_bgr(path)
            faces = extract_faces(app, img)
            faces = [f for f in faces if f["det_score"] >= min_det_score]
            if faces:
                best = max(faces, key=lambda x: x["det_score"])
                all_embs.append(best["embedding"])
                all_meta.append({
                    "path": str(path),
                    "score": float(best["det_score"]),
                    "set": path.parent.name
                })
        except Exception as e:
            continue

    if not all_embs:
        raise RuntimeError("No faces found in sets")

    # === Кластеризация (без изменений) ===
    embs = l2_normalize(np.stack(all_embs, axis=0).astype(np.float32))
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cluster_dist,
        metric="cosine",
        linkage="average"
    ).fit(embs)

    labels = clustering.labels_
    K = int(labels.max() + 1)
    ids = [f"ID{i+1:02d}" for i in range(K)]
    centroids = []
    clusters_meta = []
    coverage = {pid: {sn: 0 for sn in set_names} for pid in ids}

    for k in range(K):
        idx = np.where(labels == k)[0]
        cl_embs = embs[idx]
        centroid = l2_normalize(cl_embs.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)

        cov_per_set = {sn: 0 for sn in set_names}
        samples = []
        for i in idx:
            sn = all_meta[i]["set"]
            cov_per_set[sn] += 1
            samples.append({
                "path": all_meta[i]["path"],
                "score": all_meta[i]["score"],
                "set": sn
            })

        pid = ids[k]
        for sn, cnt in cov_per_set.items():
            coverage[pid][sn] += cnt

        clusters_meta.append({
            "id": pid,
            "count": int(len(idx)),
            "per_set": cov_per_set,
            "samples": samples
        })

    return {
        "ids": ids,
        "centroids": np.stack(centroids, axis=0).astype(np.float32),
        "clusters_meta": clusters_meta,
        "coverage": coverage,
        "set_names": set_names
    }


def match_group(app: FaceAnalysis, group_path: Path, catalog, min_det_score=0.35,
                sim_threshold=0.42, low_threshold=0.35):
    img = imread_bgr(group_path)
    faces = extract_faces(app, img)
    faces = [f for f in faces if f["det_score"] >= min_det_score]
    if not faces:
        raise RuntimeError("No faces on group image")

    group_embs = l2_normalize(np.stack([f["embedding"] for f in faces]).astype(np.float32))
    C = catalog["centroids"]
    sims = cosine_sim(group_embs, C)

    G, K = sims.shape
    assigned = [-1] * G
    assigned_sim = [0.0] * G
    for gi in range(G):
        ri = int(np.argmax(sims[gi]))
        score = float(sims[gi, ri])
        if score >= sim_threshold:
            assigned[gi] = ri
            assigned_sim[gi] = score

    present_ids = {catalog["ids"][ri] for gi, ri in enumerate(assigned) if ri >= 0}
    expected_ids = set(catalog["ids"])
    missing_ids = sorted(expected_ids - present_ids)

    extra_indices = []
    low_conf = []
    for gi in range(G):
        top_sim = float(np.max(sims[gi]))
        if top_sim < low_threshold:
            extra_indices.append(gi)
        elif low_threshold <= top_sim < sim_threshold:
            top_ri = int(np.argmax(sims[gi]))
            low_conf.append({
                "group_index": gi,
                "suggested_id": catalog["ids"][top_ri],
                "similarity": round(top_sim, 4)
            })

    labels = []
    for gi in range(G):
        if assigned[gi] >= 0:
            labels.append(f"{catalog['ids'][assigned[gi]]} ({assigned_sim[gi]:.2f})")
        else:
            top_ri = int(np.argmax(sims[gi]))
            top_sim = float(sims[gi, top_ri])
            if low_threshold <= top_sim < sim_threshold:
                labels.append(f"? {catalog['ids'][top_ri]} ({top_sim:.2f})")
            else:
                labels.append("UNKNOWN")

    return {
        "image_bgr": img,
        "faces": faces,
        "labels": labels,
        "present": sorted(present_ids),
        "missing": missing_ids,
        "extra_indices": extra_indices,
        "low_conf": low_conf,
        "similarities": sims
    }


def draw_annotated(img_bgr, faces, labels, out_path: Path):
    img = img_bgr.copy()
    for f, lab in zip(faces, labels):
        x1, y1, x2, y2 = f["bbox"]

        # 1. Увеличенная рамка (толще и ярче)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)  # ярко-жёлтая, толщина 4

        # 2. Параметры текста — крупные и контрастные
        font_scale = 1.2          # КРУПНЫЙ шрифт
        font_thickness = 3        # Жирный текст
        bg_color = (0, 0, 0)      # Чёрный фон — всегда контрастен
        text_color = (255, 255, 255)  # Белый текст

        # 3. Измеряем размер текста
        (text_w, text_h), baseline = cv2.getTextSize(
            lab, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_w = max(text_w, 180)  # Минимальная ширина фона
        text_h = text_h + baseline + 10  # Отступы сверху и снизу

        # 4. Позиция фона: СТРОГО над лицом, даже если y1 - text_h < 0
        label_y1 = max(0, y1 - text_h)
        label_y2 = max(0, y1)
        label_x1 = x1
        label_x2 = min(img.shape[1], x1 + text_w)

        # 5. Рисуем чёрный фон
        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)

        # 6. Рисуем белый текст поверх
        cv2.putText(
            img,
            lab,
            (label_x1 + 10, label_y2 - 10),  # отступы от краёв
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
    # Сохранение
    ext = out_path.suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        out_path = out_path.with_suffix(".jpg")
    ok, buf = cv2.imencode(out_path.suffix, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Failed to save annotated image")
    out_path.write_bytes(buf.tobytes())
    return str(out_path)

def compute_per_set_absences(catalog, present_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
    set_names = catalog["set_names"]
    coverage = catalog["coverage"]
    present = set(present_ids)
    all_ids = set(catalog["ids"])

    result = {}
    for sn in set_names:
        ids_in_set = {pid for pid in all_ids if coverage[pid].get(sn, 0) > 0}
        missing_here = sorted(present - ids_in_set)
        unexpected_here = sorted(ids_in_set - present)
        result[sn] = {
            "missing_present_only": missing_here,
            "unexpected_not_on_group": unexpected_here
        }
    return result