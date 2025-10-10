# setup_cx_freeze.py
from pathlib import Path
from cx_Freeze import setup, Executable
import sys

# === Метаданные приложения ===
APP_NAME = "FaceMatcher"
APP_VERSION = "1.0.0"
APP_DESC = "Face matching utility (InsightFace + OpenCV)"
ENTRY_POINT = "main.py"
TARGET_NAME = "FaceMatcher.exe"
ICON = "assets/app.ico"  # если нет — уберите параметр icon

# === Список пакетов ===
# Явно перечисляем тяжелые модули, чтобы cx_Freeze не потерял бинари
packages = [
    "cv2",
    "numpy",
    "sklearn",
    "scipy",
    "insightface",
    "onnxruntime",
    "joblib",
    "threadpoolctl",
    "utils",  # ваш модуль
]

# Исключения (то, что точно не нужно)
excludes = [
    "pytest", "unittest", "tkinter", "matplotlib", "numpy.tests",
    "scipy.tests", "sklearn.tests",
]

# Иногда scikit-learn динамически подтягивает подмодули —
# можно явно подсказать самые важные:
includes = [
    "sklearn.cluster._agglomerative",
]

# === Файлы, которые надо положить рядом с exe ===
# ОБЯЗАТЕЛЬНО: локальные модели InsightFace (см. шаг 2 ниже)
include_files = []
models_dir = Path("models")  # ожидаем models/buffalo_l/***
if models_dir.exists():
    include_files.append((str(models_dir), "models"))

# Опциональные ресурсы
for p in ["assets", "templates", "static", ".env", "config.yaml", "config.yml"]:
    if Path(p).exists():
        include_files.append((p, p))

# === Ключевые опции сборки ===
# Важно: НЕ пакуем тяжелые пакеты в zip — оставляем на диске (иначе cv2/onnxruntime ворчат)
zip_include_packages = ["*"]
zip_exclude_packages = ["cv2", "numpy", "scipy", "sklearn", "insightface", "onnxruntime"]

build_exe_options = {
    "packages": packages,
    "excludes": excludes,
    "includes": includes,
    "include_files": include_files,
    "include_msvcr": True,          # MSVC runtime
    "zip_include_packages": zip_include_packages,
    "zip_exclude_packages": zip_exclude_packages,
    # "build_exe": "build",         # можно задать свою папку сборки
    # "optimize": 1,
}

executables = [
    Executable(
        script=ENTRY_POINT,
        target_name=TARGET_NAME,
        base=None,                  # поставьте "Win32GUI" если это чистый GUI и не нужна консоль
        icon=ICON if Path(ICON).exists() else None,
    )
]

setup(
    name=APP_NAME,
    version=APP_VERSION,
    description=APP_DESC,
    options={"build_exe": build_exe_options},
    executables=executables,
)
