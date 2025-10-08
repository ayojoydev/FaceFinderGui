import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import os
import cv2
import numpy as np
import concurrent.futures

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QScrollArea, QFrame,
    QGroupBox, QProgressBar, QMessageBox, QSpinBox, QGridLayout
)
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool, QUrl
from PySide6.QtGui import QPixmap, QImage, QDesktopServices

from sklearn.cluster import AgglomerativeClustering
from insightface.app import FaceAnalysis


# ============= ЛОГИРОВАНИЕ =============
def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"log_{timestamp}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("FaceMatcherGUI")

logger = setup_logger()


# ============= КОНСТАНТЫ =============
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ============= УТИЛИТЫ =============
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


def init_face(det_size=(640, 640)) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


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


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T


# ============= ОСНОВНАЯ ЛОГИКА =============
def build_catalog_from_sets(app: FaceAnalysis, sets_root: Path, min_det_score=0.35,
                            cluster_dist=0.55, io_threads=4, progress_callback=None):
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
        progress_callback(f"Найдено изображений: {len(all_image_paths)}. Загрузка...")

    # === Параллельная загрузка изображений (I/O) ===
    def load_image_only(path: Path):
        try:
            return imread_bgr(path)
        except Exception:
            return None

    loaded_images = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_threads) as executor:
        future_to_path = {executor.submit(load_image_only, p): p for p in all_image_paths}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
            path = future_to_path[future]
            img = future.result()
            if img is not None:
                loaded_images[path] = img
            if progress_callback and i % 50 == 0:
                progress_callback(f"Загружено: {i}/{len(all_image_paths)}")

    # === Последовательная обработка лиц (CPU, insightface не thread-safe) ===
    all_embs = []
    all_meta = []
    if progress_callback:
        progress_callback("Извлечение лиц...")

    for path in all_image_paths:
        if path not in loaded_images:
            continue
        try:
            img = loaded_images[path]
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
            logger.debug(f"Пропущено {path}: {e}")
            continue

    if not all_embs:
        raise RuntimeError("No faces found in sets")

    # === Кластеризация ===
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


# ============= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ GUI =============
def create_tag(text: str, color: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"""
        background-color: {color};
        color: black;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 13px;
        margin: 2px 0;
    """)
    label.setAlignment(Qt.AlignCenter)
    return label


def numpy_to_pixmap(np_img: np.ndarray) -> QPixmap:
    rgb_image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)




# ============= РАБОЧИЙ ПОТОК =============
class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)


class AnalysisWorker(QRunnable):
    def __init__(self, group_path: Path, sets_root: Path, group_img_bgr: np.ndarray, io_threads: int):
        super().__init__()
        self.group_path = group_path
        self.sets_root = sets_root
        self.group_img_bgr = group_img_bgr
        self.io_threads = io_threads
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.progress.emit("Инициализация модели...")
            app = init_face()

            self.signals.progress.emit("Обработка сетов...")
            catalog = build_catalog_from_sets(
                app, self.sets_root,
                min_det_score=0.35,
                cluster_dist=0.55,
                io_threads=self.io_threads,
                progress_callback=self.signals.progress.emit
            )

            self.signals.progress.emit("Сопоставление с групповым фото...")
            match_result = match_group(
                app, self.group_path, catalog,
                min_det_score=0.35, sim_threshold=0.42, low_threshold=0.35
            )

            out_dir = Path("out_report")
            out_dir.mkdir(exist_ok=True)
            annotated_path = out_dir / "group_annotated.jpg"
            draw_annotated(
                match_result["image_bgr"],
                match_result["faces"],
                match_result["labels"],
                annotated_path
            )

            annotated_img = cv2.imdecode(
                np.fromfile(str(annotated_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            per_set_abs = compute_per_set_absences(catalog, match_result["present"])

            result = {
                "catalog": catalog,
                "match_result": match_result,
                "annotated_img": annotated_img,
                "annotated_path": annotated_path,
                "per_set_abs": per_set_abs
            }

            self.signals.finished.emit(result)
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            self.signals.error.emit(str(e))


# ============= ОСНОВНОЙ КЛАСС GUI =============
class FaceMatcherGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Matcher — Сверка группового фото")
        self.resize(1600, 950)

        self.group_photo_path: Optional[Path] = None
        self.group_photo_bgr: Optional[np.ndarray] = None
        self.sets_dirs: List[Path] = []
        self.thread_pool = QThreadPool()
        self.is_running = False
        self.annotated_pixmap = None
        self.init_ui()

    from PySide6.QtWidgets import QApplication

    def copy_annotated_to_clipboard(self):
        """Копирует аннотированное фото в буфер обмена"""
        if not hasattr(self, 'annotated_pixmap') or self.annotated_pixmap is None:
            # Если у нас нет pixmap — загружаем из файла
            if not self.annotated_path or not Path(self.annotated_path).exists():
                QMessageBox.warning(self, "Ошибка", "Аннотированное фото ещё не создано.")
                return
            try:
                self.annotated_pixmap = QPixmap(str(self.annotated_path))
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить фото:\n{e}")
                return

        clipboard = QApplication.clipboard()
        clipboard.setPixmap(self.annotated_pixmap)
        #QMessageBox.information(self, "Готово", "Фото скопировано в буфер обмена!\nМожно вставлять (Ctrl+V).")

    def download_annotated_photo(self):
        """Сохраняет аннотированное фото в выбранное место"""
        if not self.annotated_path or not Path(self.annotated_path).exists():
            QMessageBox.warning(self, "Ошибка", "Аннотированное фото ещё не создано.")
            return

        default_name = "group_annotated.jpg"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить аннотированное фото",
            default_name,
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;Все файлы (*)"
        )
        if save_path:
            try:
                import shutil
                shutil.copy2(self.annotated_path, save_path)
                #QMessageBox.information(self, "Успех", f"Фото сохранено:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Левый блок — фото
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        self.left_layout = left_layout
        self.select_group_btn = QPushButton("📁 Выбрать групповое фото")
        self.select_group_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        self.select_group_btn.clicked.connect(self.select_group_photo)
        left_layout.addWidget(self.select_group_btn)



        # Изначально показываем заглушку
        self._show_missing_text("Отсутствующие лица появятся здесь после анализа")
        # === Контейнер для фото + кнопки + подписей ===
        photo_container = QWidget()
        photo_layout = QVBoxLayout(photo_container)
        photo_layout.setContentsMargins(0, 0, 0, 0)
        photo_layout.setSpacing(8)

        photo_frame = QFrame()
        photo_frame.setFrameShape(QFrame.StyledPanel)
        photo_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;
                border-radius: 8px;
                background: white;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            }
        """)
        frame_layout = QVBoxLayout(photo_frame)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        self.photo_label = QLabel("Выберите групповое фото")
        self.photo_label.setAlignment(Qt.AlignCenter)
        self.photo_label.setMinimumSize(600, 550)
        self.photo_label.setStyleSheet("font-size: 14px; color: #666;")
        frame_layout.addWidget(self.photo_label)
        photo_layout.addWidget(photo_frame)

        # Кнопка скачивания (изначально скрыта)
        self.download_btn = QPushButton("💾 Скачать аннотированное фото")
        self.download_btn.setStyleSheet("font-size: 13px; padding: 6px;")
        self.download_btn.clicked.connect(self.download_annotated_photo)
        self.download_btn.setVisible(False)
        photo_layout.addWidget(self.download_btn)
        # Кнопка копирования в буффер (изначально скрыта)
        self.copy_btn = QPushButton("📋 Копировать фото в буфер")
        self.copy_btn.setStyleSheet("font-size: 13px; padding: 6px;")
        self.copy_btn.clicked.connect(self.copy_annotated_to_clipboard)
        self.copy_btn.setVisible(False)
        photo_layout.addWidget(self.copy_btn)

        # Подпись с количеством людей
        self.people_count_label = QLabel("")
        self.people_count_label.setStyleSheet("font-size: 13px; color: #2c3e50; font-weight: bold;")
        self.people_count_label.setAlignment(Qt.AlignCenter)
        photo_layout.addWidget(self.people_count_label)

        left_layout.addWidget(photo_container)

        # Прокручиваемая область для отсутствующих лиц
        self.missing_faces_scroll = QScrollArea()
        self.missing_faces_scroll.setWidgetResizable(True)
        self.missing_faces_scroll.setMaximumHeight(350)
        self.missing_faces_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.left_layout.addWidget(self.missing_faces_scroll)

        # Заготовка под scroll
        #self.missing_faces_scroll = None



        # Правый блок — управление
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(12)

        dir_btn = QPushButton("📁 Выбрать папки с сетами")
        dir_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        dir_btn.clicked.connect(self.select_sets_dirs)
        right_layout.addWidget(dir_btn)

        self.dirs_label = QLabel("Выбрано папок: 0")
        self.dirs_label.setStyleSheet("font-size: 13px; color: #555;")
        right_layout.addWidget(self.dirs_label)

        # Выбор количества потоков
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Потоков для загрузки:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 8)
        self.thread_spin.setValue(4)
        self.thread_spin.setStyleSheet("font-size: 13px;")
        thread_layout.addWidget(self.thread_spin)
        right_layout.addLayout(thread_layout)

        self.run_btn = QPushButton("🚀 Запустить анализ")
        self.run_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px;
                background: #4a86e8;
                color: white;
                border-radius: 6px;
            }
            QPushButton:disabled {
                background: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        right_layout.addWidget(self.run_btn)

        # Аналитика
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll_content = QWidget()
        self.analytics_layout = QVBoxLayout(scroll_content)
        self.analytics_layout.setSpacing(15)
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)

        # Резюме
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("""
            font-size: 15px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            background: #f0f8ff;
            border-radius: 6px;
            margin-top: 10px;
        """)
        right_layout.addWidget(self.summary_label)





        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Готово")
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)
        self.setCentralWidget(central)

    def _open_image(self, image_path: str):
        """Открывает изображение в системном просмотрщике"""
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))
        except Exception as e:
            logger.error(f"Не удалось открыть {image_path}: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть файл:\n{image_path}")

    def clear_missing_faces(self):
        """Очищает блок с отсутствующими лицами"""
        if hasattr(self, 'missing_faces_scroll') and self.missing_faces_scroll:
            # Показываем заглушку
            label = QLabel("Отсутствующие лица появятся здесь после анализа")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 13px; color: #666; padding: 20px;")
            label.setWordWrap(True)

            container = QWidget()
            layout = QVBoxLayout(container)
            layout.addWidget(label)

            self.missing_faces_scroll.setWidget(container)
            self.missing_faces_scroll.setVisible(True)

        # Сбрасываем подпись и кнопки
        self.people_count_label.setText("")
        if hasattr(self, 'download_btn'):
            self.download_btn.setVisible(False)
        if hasattr(self, 'copy_btn'):
            self.copy_btn.setVisible(False)

    def show_missing_faces(self, catalog, missing_ids):
        """Отображает миниатюры отсутствующих лиц в СЕТКЕ (grid)"""


        if not missing_ids:
            self._show_missing_text("✅ Все из сетов присутствуют на групповом фото")
            return

        missing_images = []
        for cluster in catalog["clusters_meta"]:
            if cluster["id"] in missing_ids and cluster["samples"]:
                sample = cluster["samples"][0]
                missing_images.append((cluster["id"], Path(sample["path"])))

        if not missing_images:
            self._show_missing_text("⚠️ Отсутствующие ID найдены, но нет примеров фото")
            return

        # Создаём сетку
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(15)
        grid.setAlignment(Qt.AlignTop)

        # Определяем количество колонок по ширине scroll area
        scroll_width = self.missing_faces_scroll.width()
        item_width = 180  # ширина одного элемента (120 фото + отступы)
        columns = max(1, scroll_width // item_width)

        for idx, (pid, img_path) in enumerate(missing_images[:20]):  # максимум 20
            try:
                img_bgr = imread_bgr(img_path)
                if len(img_bgr.shape) == 2:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
                elif img_bgr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

                h, w = img_bgr.shape[:2]
                if h > w:
                    start_y = (h - w) // 2
                    start_x = 0
                    crop_size = w
                else:
                    start_y = 0
                    start_x = (w - h) // 2
                    crop_size = h

                square_crop = img_bgr[start_y:start_y + crop_size, start_x:start_x + crop_size]
                resized = cv2.resize(square_crop, (120, 120), interpolation=cv2.INTER_AREA)

                pixmap = numpy_to_pixmap(resized)
                if pixmap.isNull():
                    continue

                # Элемент сетки: фото + ID
                item_widget = QWidget()
                item_layout = QVBoxLayout(item_widget)
                item_layout.setContentsMargins(0, 0, 0, 0)
                item_layout.setSpacing(6)

                img_label = QLabel()
                img_label.setFixedSize(120, 120)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setCursor(Qt.PointingHandCursor)
                img_label.setProperty("image_path", str(img_path))
                img_label.mousePressEvent = lambda e, p=str(img_path): self._open_image(p)

                id_label = QLabel(pid)
                id_label.setAlignment(Qt.AlignCenter)
                id_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #d32f2f;")

                item_layout.addWidget(img_label)
                item_layout.addWidget(id_label)

                # Добавляем в сетку: row = idx // columns, col = idx % columns
                grid.addWidget(item_widget, idx // columns, idx % columns)

            except Exception as ex:
                logger.exception(f"Ошибка при обработке {img_path}:")
                continue

        self.missing_faces_scroll.setWidget(container)

    def _show_missing_text(self, text):
        """Показывает текст в scroll area"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 13px; color: #666; padding: 20px;")
        label.setWordWrap(True)
        label.setMinimumHeight(100)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(label)

        if hasattr(self, 'missing_faces_scroll'):
            self.missing_faces_scroll.setWidget(container)

    def select_group_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите групповое фото",
            "", "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"
        )
        if path:
            self.group_photo_path = Path(path)
            try:
                self.group_photo_bgr = imread_bgr(self.group_photo_path)  # ← теперь точно массив или ошибка
                self.display_photo(self.group_photo_bgr, is_annotated=False)
                logger.info(f"Выбрано групповое фото: {self.group_photo_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки фото: {e}")
                self.photo_label.setText(f"❌ Ошибка загрузки:\n{str(e)}")
                self.group_photo_bgr = None
                self.group_photo_path = None
            self.clear_missing_faces()
            self.people_count_label.setText("")
            self.download_btn.setVisible(False)
            self.update_run_button()
            self.annotated_pixmap = None
            self.copy_btn.setVisible(False)

    def select_sets_dirs(self):
        dirs = QFileDialog.getExistingDirectory(
            self, "Выберите корневую папку с сетами",
            "", QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if dirs:
            root = Path(dirs)
            self.sets_dirs = [p for p in root.iterdir() if p.is_dir()]
            logger.info(f"Найдено сетов: {len(self.sets_dirs)} в {root}")
            self.dirs_label.setText(f"✅ Выбрано папок: {len(self.sets_dirs)} ({root.name})")
            self.clear_missing_faces()
            self.update_run_button()
            self.annotated_pixmap = None
            self.copy_btn.setVisible(False)


    def update_run_button(self):
        enabled = bool(
            self.group_photo_path and
            self.group_photo_bgr is not None and
            self.sets_dirs and
            not self.is_running
        )
        self.run_btn.setEnabled(enabled)

    def display_photo(self, img_bgr: np.ndarray, is_annotated: bool = False):
        pixmap = numpy_to_pixmap(img_bgr)
        if pixmap.isNull():
            self.photo_label.setText("❌ Ошибка отображения изображения")
            return

        # Получаем доступное пространство для фото
        max_width = self.photo_label.parentWidget().width() - 40  # отступы
        max_height = self.photo_label.height()

        # Если размеры ещё не определены — используем разумные значения
        if max_width <= 0:
            max_width = 600
        if max_height <= 0:
            max_height = 550

        scaled = pixmap.scaled(
            max_width,
            max_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.photo_label.setPixmap(scaled)
        self.photo_label.setText("")

    def run_analysis(self):
        if not self.group_photo_path or not self.sets_dirs or self.group_photo_bgr is None:
            return

        self.is_running = True
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Подготовка...")

        while self.analytics_layout.count():
            child = self.analytics_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.summary_label.setText("")
        self.clear_missing_faces()
        worker = AnalysisWorker(
            self.group_photo_path,
            self.sets_dirs[0].parent,
            self.group_photo_bgr.copy(),
            self.thread_spin.value()
        )
        worker.signals.finished.connect(self.on_analysis_finished)
        worker.signals.error.connect(self.on_analysis_error)
        worker.signals.progress.connect(self.on_progress_update)
        self.thread_pool.start(worker)

    def on_progress_update(self, message: str):
        self.progress_bar.setFormat(message)

    def update_people_count(self):
        if not hasattr(self, 'catalog') or not hasattr(self, 'match_result'):
            return

        group_count = len(self.match_result["present"])
        sets_info = []
        for set_name in self.catalog["set_names"]:
            ids_in_set = [
                pid for pid in self.catalog["ids"]
                if self.catalog["coverage"][pid].get(set_name, 0) > 0
            ]
            sets_info.append(f"{set_name}: {len(ids_in_set)}")

        # Компактный текст без переносов
        text = f"👥 На групповом фото: {group_count} человек"
        if sets_info:
            text += " | 📁 Сеты: " + ", ".join(sets_info)

        self.people_count_label.setText(text)
        self.download_btn.setVisible(True)
        self.copy_btn.setVisible(True)

    def on_analysis_finished(self, result: dict):
        self.is_running = False
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result["annotated_img"] is not None:
            self.display_photo(result["annotated_img"], is_annotated=True)
        else:
            self.photo_label.setText("❌ Ошибка загрузки аннотированного фото")

        self.catalog = result["catalog"]
        self.match_result = result["match_result"]
        self.annotated_path = result["annotated_path"]

        self.show_analytics(result["per_set_abs"])
        self.show_summary()
        self.update_people_count()
        self.show_missing_faces(self.catalog, self.match_result["missing"])
        # Сохраняем pixmap для будущего копирования
        self.annotated_pixmap = QPixmap(str(result["annotated_path"]))
        logger.info("Анализ завершён успешно.")
        self.copy_btn.setVisible(True)

    def on_analysis_error(self, error_msg: str):
        self.is_running = False
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.photo_label.setText(f"❌ Ошибка:\n{error_msg}")
        QMessageBox.critical(self, "Ошибка анализа", f"Произошла ошибка:\n{error_msg}")
        logger.error(f"Ошибка анализа: {error_msg}")

    def show_analytics(self, per_set_abs):
        COLOR_MISSING = "#ff6b6b"
        COLOR_WARNING = "#ffe66d"

        # Отсутствуют на групповом фото
        missing_group = QGroupBox("🔴 Отсутствуют на групповом фото")
        missing_group.setStyleSheet("font-weight: bold; font-size: 14px;")
        missing_layout = QVBoxLayout()
        if self.match_result["missing"]:
            for mid in self.match_result["missing"]:
                tag = create_tag(mid, COLOR_MISSING)
                missing_layout.addWidget(tag)
        else:
            missing_layout.addWidget(QLabel("— Все присутствуют"))
        missing_group.setLayout(missing_layout)
        self.analytics_layout.addWidget(missing_group)

        # Лишние на групповом фото
        extra_group = QGroupBox("🟡 Лишние на групповом фото")
        extra_group.setStyleSheet("font-weight: bold; font-size: 14px;")
        extra_layout = QVBoxLayout()
        extra_count = len(self.match_result["extra_indices"])
        if extra_count > 0:
            extra_layout.addWidget(QLabel(f"{extra_count} лиц(а) не идентифицированы"))
            for _ in range(extra_count):
                tag = create_tag("UNKNOWN", COLOR_WARNING)
                extra_layout.addWidget(tag)
        else:
            extra_layout.addWidget(QLabel("— Нет лишних лиц"))
        extra_group.setLayout(extra_layout)
        self.analytics_layout.addWidget(extra_group)

        # По сетам
        for set_name, data in per_set_abs.items():
            set_group = QGroupBox(f"📁 Сет: {set_name}")
            set_group.setStyleSheet("font-weight: bold; font-size: 14px;")
            set_layout = QVBoxLayout()

            if data["missing_present_only"]:
                set_layout.addWidget(QLabel("Отсутствуют в этом сете (но есть на группе):"))
                for mid in data["missing_present_only"]:
                    tag = create_tag(mid, COLOR_MISSING)
                    set_layout.addWidget(tag)
            else:
                set_layout.addWidget(QLabel("✅ Все из группы присутствуют в этом сете"))

            if data["unexpected_not_on_group"]:
                set_layout.addWidget(QLabel("Есть в сете, но отсутствуют на группе:"))
                for uid in data["unexpected_not_on_group"]:
                    tag = create_tag(uid, COLOR_WARNING)
                    set_layout.addWidget(tag)

            set_group.setLayout(set_layout)
            self.analytics_layout.addWidget(set_group)

    def show_summary(self):
        total = len(self.catalog["ids"])
        missing = len(self.match_result["missing"])
        extra = len(self.match_result["extra_indices"])
        text = f"📊 Всего: {total} | Отсутствуют: {missing} | Лишние: {extra}"
        self.summary_label.setText(text)


# ============= ЗАПУСК =============
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceMatcherGUI()
    window.show()
    sys.exit(app.exec())