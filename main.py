from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from typing import List, Dict
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QScrollArea, QFrame,
    QGroupBox, QProgressBar, QMessageBox, QSpinBox, QGridLayout
)
from PySide6.QtCore import Qt, QObject, Signal,  QUrl
from PySide6.QtGui import QPixmap, QImage, QDesktopServices

# Логика анализа
from face_matcher import (
    build_catalog_from_sets,
    match_group,
    draw_annotated,
    compute_per_set_absences
)

from utils import imread_bgr, ALLOWED_EXTS

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


# ============= ОСНОВНОЙ КЛАСС GUI =============
class FaceMatcherGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Matcher — Сверка группового фото")
        self.resize(1600, 950)

        self.group_photo_path: Optional[Path] = None
        self.group_photo_bgr: Optional[np.ndarray] = None
        self.sets_dirs: List[Path] = []
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
            except Exception as e:
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

        # === ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ ===
        # Очищаем аналитикуS
        while self.analytics_layout.count():
            child = self.analytics_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.summary_label.setText("")
        self.clear_missing_faces()  # очищает миниатюры
        # =================================

        # Блокируем интерфейс
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Инициализация модели...")
        QApplication.processEvents()  # Обновляем UI

        try:
            from face_matcher import init_face  # импортируем здесь

            # 1. Инициализация
            self.progress_bar.setFormat("Инициализация модели...")
            QApplication.processEvents()
            app = init_face()

            # 2. Построение каталога
            self.progress_bar.setFormat("Обработка сетов...")
            QApplication.processEvents()
            catalog = build_catalog_from_sets(
                app, self.sets_dirs[0].parent,
                min_det_score=0.35,
                cluster_dist=0.55,
                progress_callback=lambda msg: (
                    setattr(self.progress_bar, 'format', msg),
                    QApplication.processEvents()
                )
            )

            # 3. Матчинг группы
            self.progress_bar.setFormat("Сопоставление с групповым фото...")
            QApplication.processEvents()
            match_result = match_group(
                app, self.group_photo_path, catalog,
                min_det_score=0.35, sim_threshold=0.42, low_threshold=0.35
            )

            # 4. Сохранение аннотированного фото
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

            # 5. Обновление UI
            if annotated_img is not None:
                self.display_photo(annotated_img, is_annotated=True)
            else:
                self.photo_label.setText("❌ Ошибка загрузки аннотированного фото")

            self.catalog = catalog
            self.match_result = match_result
            self.annotated_path = annotated_path
            self.annotated_pixmap = QPixmap(str(annotated_path))

            self.show_analytics(per_set_abs)
            self.show_summary()
            self.update_people_count()
            self.show_missing_faces(catalog, match_result["missing"])


        except Exception as e:
            import traceback

            self.photo_label.setText(f"❌ Ошибка:\n{str(e)}")
            QMessageBox.critical(self, "Ошибка анализа", f"Произошла ошибка:\n{str(e)}")

        finally:
            # Разблокируем интерфейс
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.download_btn.setVisible(True)
            self.copy_btn.setVisible(True)

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
        self.copy_btn.setVisible(True)

    def on_analysis_error(self, error_msg: str):
        self.is_running = False
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.photo_label.setText(f"❌ Ошибка:\n{error_msg}")
        QMessageBox.critical(self, "Ошибка анализа", f"Произошла ошибка:\n{error_msg}")

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
        # По сетам
        for set_name, data in per_set_abs.items():
            # Считаем, сколько уникальных людей в этом сете
            ids_in_set = [
                pid for pid in self.catalog["ids"]
                if self.catalog["coverage"][pid].get(set_name, 0) > 0
            ]
            set_count = len(ids_in_set)

            set_group = QGroupBox(f"📁 Сет: {set_name} ({set_count} чел.)")
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