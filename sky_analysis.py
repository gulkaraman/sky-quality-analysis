import os
import sys
import io
import json
import math
import random
import shutil
import logging
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import importlib.metadata as importlib_metadata  # Python 3.8+
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:
    import seaborn as sns
except ImportError:  # Seaborn opsiyonel
    sns = None

from PIL import Image, ExifTags
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
)

# TensorFlow opsiyonel: varsa ML kısmı tam çalışır, yoksa sadece CV/rapor kısmı
try:
    import tensorflow as tf
    from tensorflow import keras  # tek giriş noktası: keras.layers / keras.applications / keras.callbacks ...
    # EfficientNet sınıfını doğrudan keras.applications üzerinden al; alt modülü ayrıca import etme
    EfficientNetB0 = keras.applications.EfficientNetB0
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    keras = None
    EfficientNetB0 = None
    HAS_TENSORFLOW = False


# ==========================
# AYARLANABİLİR PARAMETRELER
# ==========================

# Ana veri klasörü (kullanıcıdan verilen)
DATA_DIR = ""

# Eğer yukarıdaki yoksa, bir üst dizinde ASC_Data aramayı dene
FALLBACK_PARENT_SEARCH = True

# Görüntü yeniden boyutlandırma
IMAGE_SIZE = (224, 224)  # (height, width)

# Model eğitim parametreleri
BATCH_SIZE = 16
EPOCHS = 300
SEED = 42

# Ay filtresi (örn "2021-10" veya None). None ise mevcut aylar raporlanır.
MONTH_FILTER: Optional[str] = None  # "YYYY-MM" formatında string veya None
# MONTH_FILTER None iken en yeni ayı otomatik seçmek için True yapın
AUTO_MONTH_LATEST = False

# Gündüz/gece ayrımı: ortalama parlaklık (V) bu eşiğin altındaysa gece sayılır
NIGHT_V_MEAN_THRESH = 30

# Gece görüntüleri: (i) unknown_night + cloudiness NaN, (ii) clear_night/overcast_night. Şu an (i) kullanılıyor.
NIGHT_POLICY = "unknown_night"  # "unknown_night" | "clear_overcast_night"

# Raporlarda gece görüntülerini özet/grafiklere dahil et
INCLUDE_NIGHT_IN_SUMMARY = False

# ML eğitimini varsayılan olarak kapat; --train-ml veya TRAIN_ML=True ile açılır
TRAIN_ML = False
# ML etiketi: "folder" (klasör adı) veya "category" (clear/partly_cloudy/overcast, kural-tabanlı pseudo-label)
ML_LABEL_COLUMN = "category"

# Çıktı klasörü
OUTPUT_DIR = "outputs"

# Global kontrol bayrakları (CLI ile güncellenecek)
NO_PROGRESS = False
FORCE_CPU = False
LOG_EVERY_BATCHES = 0


# ==========================
# LOGGING / YARDIMCI FONK.
# ==========================

logger = logging.getLogger("sky_analysis")


def setup_logging(level_str: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure global logger once; optional file handler."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler (stdout)
    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    ):
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Optional file handler
    if log_file:
        log_path = os.path.abspath(log_file)
        if not any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == log_path
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
            fh.setFormatter(fmt)
            logger.addHandler(fh)


def log(msg: str, level: int = logging.INFO) -> None:
    """Unified logging helper."""
    logger.log(level, msg)


def log_step(title: str) -> None:
    """Pretty step banner for console; kolay okunur akış."""
    line = "=" * 80
    log("\n" + line)
    log(f">>> {title}")
    log(line)


@contextmanager
def timed_step(title: str):
    """Context manager that logs step start/end with elapsed time."""
    log_step(title)
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        log(f"{title} süresi: {elapsed:.2f} s")


def set_global_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TENSORFLOW:
        tf.random.set_seed(seed)


def ensure_output_dir(path: str) -> None:
    """Create outputs directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _safe_get_package_version(name: str) -> Optional[str]:
    """Helper to get package version; returns None if not installed."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def print_system_info(output_dir: str) -> None:
    """Print and save system and environment information."""
    info: Dict[str, Dict[str, object]] = {}

    # OS / platform
    info["os"] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # CPU / RAM / disk (psutil opsiyonel)
    cpu_count = os.cpu_count()
    info["cpu"] = {"cores": cpu_count}
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        du = psutil.disk_usage(output_dir)
        info["ram"] = {
            "total": vm.total,
            "available": vm.available,
            "percent": vm.percent,
        }
        info["disk"] = {
            "total": du.total,
            "used": du.used,
            "free": du.free,
            "percent": du.percent,
        }
    except Exception:
        log("psutil bulunamadı veya kullanılamadı; RAM/disk bilgisi sınırlı.", logging.WARNING)
        info.setdefault("ram", {}).update({"total": None})
        info.setdefault("disk", {}).update({"total": None})

    # Python
    info["python"] = {
        "executable": sys.executable,
        "version": sys.version,
    }

    # Paket sürümleri
    pkgs = ["tensorflow", "keras", "numpy", "opencv-python", "pandas", "scikit-learn", "matplotlib"]
    info["packages"] = {name: _safe_get_package_version(name) for name in pkgs}

    # Env vars
    env_keys = ["CUDA_VISIBLE_DEVICES", "TF_ENABLE_ONEDNN_OPTS", "TF_CPP_MIN_LOG_LEVEL"]
    info["env"] = {k: os.environ.get(k) for k in env_keys}

    # Log to console
    log_step("Sistem Bilgisi")
    log(json.dumps(info, indent=2, ensure_ascii=False))

    # Save to JSON
    try:
        ensure_output_dir(output_dir)
        sys_path = os.path.join(output_dir, "system_info.json")
        with open(sys_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        log(f"Sistem bilgisi kaydedildi: {sys_path}")
    except Exception as e:
        log(f"Sistem bilgisi JSON yazılamadı: {e}", logging.WARNING)


def resolve_data_dir(data_dir: str) -> str:
    """Resolve the actual data directory with fallback search.

    1) Use given DATA_DIR if exists.
    2) Else, search for 'ASC_Data' in parent directory if enabled.
    3) If still not found, raise RuntimeError (fail fast).
    """
    if os.path.isdir(data_dir):
        return data_dir

    if FALLBACK_PARENT_SEARCH:
        parent = os.path.dirname(os.path.abspath(data_dir))
        candidate = os.path.join(parent, "ASC_Data")
        if os.path.isdir(candidate):
            return candidate

    msg = (
        f"Veri klasörü bulunamadı: {data_dir}\n"
        "Lütfen DATA_DIR yolunu kontrol edin veya 'ASC_Data' klasörünü bir üst dizine yerleştirin."
    )
    raise RuntimeError(msg)


def get_exif_datetime(img_path: str) -> Optional[datetime]:
    """Extract capture datetime from image EXIF metadata if available."""
    try:
        img = Image.open(img_path)
        exif = img._getexif()
        if not exif:
            return None
        exif_data = {
            ExifTags.TAGS.get(k, k): v
            for k, v in exif.items()
            if k in ExifTags.TAGS
        }
        # Common EXIF datetime tags
        for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if key in exif_data:
                dt_str = exif_data[key]
                try:
                    # Typical format: "2021:10:05 21:30:00"
                    return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                except Exception:
                    continue
    except Exception:
        return None
    return None


def parse_datetime_from_filename(fname: str) -> Optional[datetime]:
    """Best-effort datetime parsing from filename using common patterns."""
    name = os.path.splitext(os.path.basename(fname))[0]

    # Try patterns like: YYYYMMDD_HHMMSS, YYYY-MM-DD_HH-MM-SS, etc.
    patterns = [
        "%Y%m%d_%H%M%S",
        "%Y-%m-%d_%H-%M-%S",
        "%Y-%m-%d-%H-%M-%S",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d %H-%M-%S",
        "%Y-%m-%d %H:%M:%S",
    ]

    for p in patterns:
        try:
            return datetime.strptime(name, p)
        except Exception:
            continue

    # If there are digits only, try to slice plausible segments
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 14:
        try:
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        except Exception:
            pass

    return None


def get_image_datetime(path: str) -> Optional[datetime]:
    """Get image datetime from EXIF or filename."""
    dt = get_exif_datetime(path)
    if dt is not None:
        return dt
    return parse_datetime_from_filename(path)


def compute_mean_brightness(image_bgr: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> float:
    """Compute mean brightness from BGR image (V channel of HSV). Optionally only over ROI."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    if roi_mask is not None and roi_mask.any():
        # Sadece ROI içindeki piksellerin ortalaması (ROI 0/1 binary)
        v_roi = v.astype(np.float64)[roi_mask.astype(bool)]
        return float(v_roi.mean()) if len(v_roi) > 0 else 0.0
    return float(v.mean())


def detect_fisheye_circle_hough(gray: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int]]:
    """Fisheye/all-sky dairesel bölgeyi HoughCircles ile tespit et. (cx, cy, r) veya None döner."""
    # Kenar artefaktlarını azaltmak için hafif blur
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    min_side = min(h, w)
    min_dist = max(1, min_side // 2)
    min_r = max(10, int(0.30 * min_side))
    max_r = max(min_r + 1, int(0.60 * min_side))
    # HoughCircles parametreleri: dp, minDist, param1, param2, minRadius, maxRadius
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None or len(circles) == 0:
        return None
    circles = np.uint16(np.around(circles))
    # En büyük yarıçapı seç; merkez görüntü içinde olsun
    best = None
    for cx, cy, r in circles[0]:
        if 0 <= cx < w and 0 <= cy < h and r >= min_r and r <= max_r:
            if best is None or r > best[2]:
                best = (int(cx), int(cy), int(r))
    if best is None:
        return None
    cx, cy, r = best
    # Yarıçapı %2-5 küçült (kenar artefaktı azaltma)
    r = max(1, int(r * 0.95))
    return (cx, cy, r)


def _contour_circularity(contour: np.ndarray) -> float:
    """Circularity = 4*pi*area / perimeter^2; daire için 1'e yakın."""
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perim * perim)


def _detect_sky_roi_contour_fallback(gray: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int]]:
    """Otsu + kontur ile fallback: circularity + alan + yarıçap aralığı ile en uygun daireyi seç."""
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OpenCV sürümüne göre 2 veya 3 değer dönebilir
    cnt_result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnt_result[0] if len(cnt_result) == 2 else cnt_result[1]
    if not contours:
        return None
    min_side = min(h, w)
    min_r, max_r = int(0.25 * min_side), int(0.55 * min_side)
    center_x, center_y = w / 2.0, h / 2.0
    best_score, best_circle = -1.0, None
    for c in contours:
        area = cv2.contourArea(c)
        if area < (min_side * 0.1) ** 2:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        if radius < min_r or radius > max_r:
            continue
        circ = _contour_circularity(c)
        # Merkeze yakınlık bonusu
        dist_from_center = math.hypot(x - center_x, y - center_y) / min_side
        score = circ * (area / (min_side * min_side)) * (1.0 - 0.5 * min(1.0, dist_from_center))
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), max(1, int(radius * 0.95)))
    return best_circle


def detect_sky_roi(image_bgr: np.ndarray) -> np.ndarray:
    """Sky ROI mask: sadece 0/1 uint8. Önce Hough, sonra kontur fallback, sonra tüm frame."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    circle = detect_fisheye_circle_hough(gray, h, w)
    if circle is None:
        circle = _detect_sky_roi_contour_fallback(gray, h, w)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    if circle is not None:
        cx, cy, r = circle
        if 0 <= cx < w and 0 <= cy < h and r > min(h, w) * 0.15:
            cv2.circle(roi_mask, (cx, cy), r, 1, thickness=-1)
    if roi_mask.sum() == 0:
        roi_mask[:, :] = 1
    strip = int(0.1 * h)
    roi_mask[h - strip : h, :] = 0
    return roi_mask


def build_cloud_mask(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    ignore_sun_glare: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build cloud binary mask (0/1 only). R/B + saturation + Otsu; morph on 0-255 then back to 0/1."""
    img = image_bgr.astype(np.float32) / 255.0
    b, r = img[:, :, 0], img[:, :, 2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1].astype(np.float32) / 255.0
    v_chan = hsv[:, :, 2].astype(np.float32) / 255.0
    eps = 1e-3
    rb_ratio = r / (b + eps)
    rb_thresh, sat_thresh = 1.1, 0.25
    mask_rb = ((rb_ratio > rb_thresh).astype(np.uint8))  # 0/1
    mask_sat = ((s_chan < sat_thresh).astype(np.uint8))
    v_uint8 = (v_chan * 255).astype(np.uint8)
    _, v_otsu = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu = (v_otsu == 255).astype(np.uint8)
    combined_score = mask_rb + mask_sat + mask_otsu
    cloud_mask = (combined_score >= 2).astype(np.uint8)  # 0/1

    final_roi = (roi_mask.copy().astype(np.uint8) & 1)  # Garanti 0/1
    if ignore_sun_glare:
        glare_mask = (v_chan > 0.98).astype(np.uint8)
        final_roi = (final_roi & (1 - glare_mask)).astype(np.uint8)
        cloud_mask = (cloud_mask & (1 - glare_mask)).astype(np.uint8)
    cloud_mask = (cloud_mask & final_roi).astype(np.uint8)

    # Morfoloji: OpenCV 0/255 bekler; işlem sonrası tekrar 0/1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cloud_255 = (cloud_mask * 255).astype(np.uint8)
    cloud_255 = cv2.morphologyEx(cloud_255, cv2.MORPH_OPEN, kernel)
    cloud_255 = cv2.morphologyEx(cloud_255, cv2.MORPH_CLOSE, kernel)
    cloud_mask = (cloud_255 > 0).astype(np.uint8)
    return cloud_mask, final_roi


def compute_cloudiness(
    image_bgr: np.ndarray,
) -> Tuple[float, str, bool, np.ndarray, np.ndarray]:
    """Bulutluluk: önce ROI, ROI üzerinde parlaklık ile gündüz/gece; gündüzde R/B+S+Otsu, gecede NIGHT_POLICY."""
    roi_mask = detect_sky_roi(image_bgr)
    # Parlaklığı ROI içinde hesapla (gündüz/gece ayrımı daha doğru)
    mean_bright = compute_mean_brightness(image_bgr, roi_mask)
    is_night = mean_bright < NIGHT_V_MEAN_THRESH

    cloud_mask, final_roi = build_cloud_mask(image_bgr, roi_mask)
    sky_pixels = int(final_roi.sum())
    cloud_pixels = int(cloud_mask.sum())

    if sky_pixels == 0:
        cloudiness = math.nan
    else:
        cloudiness = cloud_pixels / float(sky_pixels)

    if is_night:
        # Seçenek (i): gece = unknown_night, cloudiness NaN veya mevcut hesapla ama kategori ayrı
        if NIGHT_POLICY == "unknown_night":
            category = "unknown_night"
            cloudiness = math.nan
        else:
            pct = cloudiness * 100 if not math.isnan(cloudiness) else 0
            category = "clear_night" if pct <= 30 else "overcast_night"
    else:
        pct = cloudiness * 100 if not math.isnan(cloudiness) else math.nan
        if math.isnan(pct):
            category = "unknown"
        elif pct <= 10:
            category = "clear"
        elif pct <= 30:
            category = "partly_cloudy"
        else:
            category = "overcast"

    return cloudiness, category, is_night, final_roi, cloud_mask


def discover_images(data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """Discover images in data_dir and build dataset profile DataFrame.

    Returns:
        df: DataFrame with columns:
            filepath, folder, extension, width, height,
            datetime, mean_brightness, is_night
        corrupted_files: list of paths that could not be opened
    Ayrıca, desteklenmeyen uzantıları da sayıp raporlar (modele eklemez)."""
    # Desteklenen görüntü uzantıları
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp")
    filepaths: List[str] = []
    all_files: List[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            full = os.path.join(root, f)
            all_files.append(full)
            if f.lower().endswith(exts):
                filepaths.append(full)

    # Farklı uzantıları konsola raporla
    ext_counts: Dict[str, int] = {}
    for p in all_files:
        ext = os.path.splitext(p)[1].lower() or "<no_ext>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    log_step("Veri klasörü taraması")
    log(f"Toplam dosya sayısı: {len(all_files)}")
    log(f"Görüntü olarak işlenecek uzantılar: {exts}")
    log(f"Uzantı dağılımı (tüm dosyalar): {ext_counts}")

    records = []
    corrupted = []

    for path in filepaths:
        rel_folder = os.path.basename(os.path.dirname(path))
        ext = os.path.splitext(path)[1].lower()
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imread returned None")
            h, w = img.shape[:2]
            dt = get_image_datetime(path)
            mean_bright = compute_mean_brightness(img)  # Keşif aşamasında tüm frame
            is_night = mean_bright < NIGHT_V_MEAN_THRESH
            records.append(
                {
                    "filepath": path,
                    "folder": rel_folder,
                    "extension": ext,
                    "width": w,
                    "height": h,
                    "datetime": dt,
                    "mean_brightness": mean_bright,
                    "is_night": is_night,
                }
            )
        except Exception:
            corrupted.append(path)

    df = pd.DataFrame(records)
    # MONTH_FILTER merkezi: datetime kolonunu garanti et, NaT sayısını raporla
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        nat_count = df["datetime"].isna().sum()
        if nat_count > 0:
            print(f"  (Tarih parse edilemeyen görüntü sayısı: {nat_count})")
    return df, corrupted


def profile_dataset(df: pd.DataFrame, corrupted: List[str]) -> pd.DataFrame:
    """Create dataset profile and print summary tables."""
    print("\n=== Veri Seti Özeti ===")
    print(f"Toplam geçerli görüntü: {len(df)}")
    print(f"Bozuk / okunamayan görüntü: {len(corrupted)}")

    # Klasör başına dosya sayısı
    print("\nKlasör başına dosya sayısı:")
    if not df.empty and "folder" in df:
        print(df["folder"].value_counts())

    # Uzantı dağılımı
    print("\nUzantı dağılımı:")
    if not df.empty and "extension" in df:
        print(df["extension"].value_counts())

    # Çözünürlük dağılımı
    print("\nÇözünürlük dağılımı (ilk 10):")
    if not df.empty and {"width", "height"}.issubset(df.columns):
        res_counts = df.groupby(["width", "height"]).size().reset_index(name="count")
        print(res_counts.sort_values("count", ascending=False).head(10))

    # Datetime min/max ve yoğunluk; MONTH_FILTER None ise mevcut ayları listele
    if "datetime" in df and df["datetime"].notnull().any():
        dt_series = df["datetime"].dropna()
        print("\nTarih aralığı:")
        print(f"Min: {dt_series.min()}, Max: {dt_series.max()}")
        df["date"] = df["datetime"].dt.date
        df["month"] = df["datetime"].dt.to_period("M").astype(str)
        print("\nAy bazında yoğunluk:")
        print(df["month"].value_counts().sort_index())
        available_months = sorted(df["month"].dropna().unique().tolist())
        print(f"Mevcut aylar (YYYY-MM): {available_months}")
    else:
        print("\nTarih bilgisi bulunamadı.")

    return df


def compute_cloudiness_for_all(
    df: pd.DataFrame,
    debug_info: Optional[Dict] = None,
) -> pd.DataFrame:
    """Tüm görüntüler için bulutluluk; debug_info verilirse sky_pixels==0 ve ROI unique değerlerini topla."""
    results = []
    if debug_info is not None:
        debug_info["sky_pixels_zero_paths"] = []
        debug_info["roi_unique_values"] = set()

    for _, row in df.iterrows():
        path = row["filepath"]
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imread returned None")
            cloudiness, category, is_night_flag, final_roi, cloud_mask = compute_cloudiness(img)
            if debug_info is not None:
                debug_info["roi_unique_values"].update(np.unique(final_roi).tolist())
                if int(final_roi.sum()) == 0:
                    debug_info["sky_pixels_zero_paths"].append(path)
            results.append(
                {
                    "filepath": path,
                    "datetime": row.get("datetime"),
                    "folder": row.get("folder"),
                    "cloudiness": cloudiness,
                    "category": category,
                    "is_night": is_night_flag,
                }
            )
        except Exception as e:
            print(f"Bulutluluk hesaplanamadı: {path} ({e})")

    return pd.DataFrame(results)


def summarize_cloudiness(cloud_df: pd.DataFrame) -> None:
    """Aylık özet: gündüz için clear/partly/overcast; isteğe bağlı gece ayrı."""
    df = cloud_df.copy()
    if "datetime" in df and df["datetime"].notnull().any() and MONTH_FILTER:
        df = df[df["datetime"].dt.to_period("M").astype(str) == MONTH_FILTER]
    df_day = df[~df["is_night"]].copy() if "is_night" in df else df.copy()
    df_night = df[df["is_night"]].copy() if "is_night" in df else pd.DataFrame()

    clear_pct = (df_day["category"] == "clear").mean() * 100 if not df_day.empty else 0
    partly_pct = (df_day["category"] == "partly_cloudy").mean() * 100 if not df_day.empty else 0
    overcast_pct = (df_day["category"] == "overcast").mean() * 100 if not df_day.empty else 0

    print("\n=== Aylık Gözlem Özeti (Gündüz) ===")
    if MONTH_FILTER:
        print(f"Ay filtresi: {MONTH_FILTER}")
    print(f"Bu ayın yaklaşık %{clear_pct:.1f}'inde gökyüzü tamamen açıktı.")
    print(f"%{partly_pct:.1f}'inde parçalı bulutlu, %{overcast_pct:.1f}'inde gözlem yapılamaz (kapalı).")
    if INCLUDE_NIGHT_IN_SUMMARY and not df_night.empty:
        print(f"Gece görüntü sayısı: {len(df_night)} (NIGHT_POLICY={NIGHT_POLICY})")


def plot_time_series_and_histograms(
    cloud_df: pd.DataFrame, output_dir: str
) -> None:
    """Generate required plots: daily changes, histograms, hourly averages, pie charts."""
    df = cloud_df.copy()
    if "datetime" not in df or not df["datetime"].notnull().any():
        return

    if MONTH_FILTER:
        df = df[df["datetime"].dt.to_period("M").astype(str) == MONTH_FILTER]
    df = df.dropna(subset=["cloudiness"])
    df_day = df[~df["is_night"]]

    if not df_day.empty:
        df_day["date"] = df_day["datetime"].dt.date

        # Günlük değişim grafiği
        daily = df_day.groupby("date")["cloudiness"].mean().reset_index()
        plt.figure(figsize=(10, 4))
        plt.plot(daily["date"], daily["cloudiness"] * 100, marker="o")
        plt.ylabel("Ortalama Bulutluluk (%)")
        plt.xlabel("Tarih")
        plt.title("Günlük Bulutluluk Değişimi")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "daily_cloudiness.png"))
        plt.close()

        # Saatlik ortalama bulutluluk
        df_day["hour"] = df_day["datetime"].dt.hour
        hourly = df_day.groupby("hour")["cloudiness"].mean().reset_index()
        plt.figure(figsize=(8, 4))
        plt.bar(hourly["hour"], hourly["cloudiness"] * 100)
        plt.xlabel("Saat")
        plt.ylabel("Ortalama Bulutluluk (%)")
        plt.title("Saatlik Ortalama Bulutluluk")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hourly_cloudiness.png"))
        plt.close()

    # Pasta grafik (kategori oranları)
    cat_counts = df_day["category"].value_counts()
    if not cat_counts.empty:
        plt.figure(figsize=(5, 5))
        plt.pie(
            cat_counts.values,
            labels=cat_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Açık / Parçalı / Kapalı Oranları")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cloud_category_pie.png"))
        plt.close()

    # Bulutluluk histogramı
    if not df_day.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(df_day["cloudiness"] * 100, bins=30, color="skyblue", edgecolor="k")
        plt.xlabel("Bulutluluk (%)")
        plt.ylabel("Frekans")
        plt.title("Bulutluluk Histogramı")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cloudiness_histogram.png"))
        plt.close()


def save_segmentation_examples(
    cloud_df: pd.DataFrame, output_dir: str, num_examples: int = 4
) -> None:
    """Dört panel: Orijinal / roi_raw / final_roi / cloud_mask (final_roi glare çıkarılmış ROI)."""
    df = cloud_df.copy()
    # Geçerli cloudiness olan veya gece olan örnek al
    df = df[df["cloudiness"].notna() | (df.get("category") == "unknown_night")]
    if df.empty:
        df = cloud_df.head(min(num_examples, len(cloud_df)))
    sample_df = df.sample(min(num_examples, len(df)), random_state=SEED)
    for idx, (i, row) in enumerate(sample_df.iterrows()):
        path = row["filepath"]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        roi_raw = detect_sky_roi(img)
        cloud_mask, final_roi = build_cloud_mask(img, roi_raw)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Orijinal")
        axes[0].axis("off")
        axes[1].imshow(roi_raw, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("roi_raw")
        axes[1].axis("off")
        axes[2].imshow(final_roi, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("final_roi (glare çıkarılmış)")
        axes[2].axis("off")
        axes[3].imshow(cloud_mask, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("cloud_mask")
        axes[3].axis("off")
        plt.tight_layout()
        fname = os.path.join(output_dir, f"segmentation_example_{idx}.png")
        plt.savefig(fname)
        plt.close()


def prepare_splits(
    cloud_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Stratified train/val/test; etiket ML_LABEL_COLUMN (category veya folder)."""
    df = cloud_df.copy()
    df = df.dropna(subset=["cloudiness"])
    label_col = ML_LABEL_COLUMN
    if label_col not in df.columns:
        label_col = "folder"
    df = df[df[label_col].notnull()]

    df = df[~df["is_night"]].copy()
    # category ile eğitirken sadece clear/partly_cloudy/overcast kullan (pseudo-label)
    if label_col == "category":
        df = df[df["category"].isin(["clear", "partly_cloudy", "overcast"])]

    if MONTH_FILTER and "datetime" in df and df["datetime"].notnull().any():
        df = df[df["datetime"].dt.to_period("M").astype(str) == MONTH_FILTER]

    if df.empty:
        raise RuntimeError("Eğitim için uygun görüntü bulunamadı (filtreleri kontrol edin).")

    min_per_class = 2
    counts = df[label_col].value_counts()
    valid_labels = counts[counts >= min_per_class].index.tolist()
    df = df[df[label_col].isin(valid_labels)]
    if len(valid_labels) < 2:
        raise RuntimeError("Stratified split için en az 2 sınıf ve her sınıfta en az 2 örnek gerekir.")

    labels = sorted(valid_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y = df[label_col].values
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=y)
    y_train_val = train_val_df[label_col].values
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=SEED, stratify=y_train_val
    )
    return train_df, val_df, test_df, label_to_idx


def build_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_to_idx: Dict[str, int],
):
    """Keras generator: EfficientNet için preprocess_input kullan, rescale kaldırıldı (0-255 -> preprocess)."""
    if not HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow bulunamadı, generator oluşturulamıyor.")
    label_col = ML_LABEL_COLUMN if ML_LABEL_COLUMN in train_df.columns else "folder"
    classes = sorted(label_to_idx.keys())

    def df_to_generator(df: pd.DataFrame, augment: bool):
        datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.efficientnet.preprocess_input,
            horizontal_flip=True if augment else False,
            rotation_range=15 if augment else 0,
            zoom_range=0.15 if augment else 0.0,
            width_shift_range=0.1 if augment else 0.0,
            height_shift_range=0.1 if augment else 0.0,
            brightness_range=(0.85, 1.15) if augment else None,
        )
        gen = datagen.flow_from_dataframe(
            df,
            x_col="filepath",
            y_col=label_col,
            classes=classes,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=augment,
            seed=SEED,
        )
        return gen

    train_gen = df_to_generator(train_df, augment=True)
    val_gen = df_to_generator(val_df, augment=False)
    test_gen = df_to_generator(test_df, augment=False)
    return train_gen, val_gen, test_gen


def build_model(num_classes: int, backbone_name: str = "EfficientNetB0"):
    """Build transfer learning model using EfficientNet backbone.

    backbone_name: 'EfficientNetB0' veya 'EfficientNetB3' gibi keras.applications içindeki isim.
    """
    if not HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow bulunamadı, model oluşturulamıyor.")

    try:
        backbone_fn = getattr(keras.applications, backbone_name)
    except AttributeError:
        raise RuntimeError(f"Desteklenmeyen backbone: {backbone_name}")

    # Önce ImageNet ağırlıklarıyla dene; internet yoksa random init'e düş
    try:
        backbone = backbone_fn(
            include_top=False,
            weights="imagenet",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        )
    except Exception as e:
        log(
            f"[WARN] {backbone_name} için ImageNet ağırlıkları indirilemedi "
            f"veya yüklenemedi ({e}). Random init (weights=None) ile devam ediliyor."
        )
        backbone = backbone_fn(
            include_top=False,
            weights=None,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        )
    backbone.trainable = False  # İlk aşamada dondur

    inputs = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class EpochLogger(keras.callbacks.Callback):
    """Her epoch başı/sonunda metrik + süre + LR loglar."""

    def __init__(self):
        super().__init__()
        self._epoch_start_time: Optional[float] = None

    def on_epoch_begin(self, epoch, logs=None):
        total = self.params.get("epochs")
        self._epoch_start_time = time.time()
        log(f"----- Epoch {epoch + 1}/{total} başlıyor -----")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def _fmt(key):
            v = logs.get(key)
            return f"{v:.4f}" if isinstance(v, (float, int)) else "NA"

        elapsed = time.time() - (self._epoch_start_time or time.time())
        # LR'yi optimizer'dan güvenli al
        lr_val = None
        try:
            lr = self.model.optimizer.learning_rate  # type: ignore[attr-defined]
            if hasattr(lr, "numpy"):
                lr_val = float(lr.numpy())
            else:
                lr_val = float(keras.backend.get_value(lr))
        except Exception:
            lr_val = None
        lr_str = f"{lr_val:.6f}" if isinstance(lr_val, (float, int)) else "NA"

        msg = (
            f"Epoch {epoch + 1} bitti | "
            f"loss={_fmt('loss')}, acc={_fmt('accuracy')}, "
            f"val_loss={_fmt('val_loss')}, val_acc={_fmt('val_accuracy')}, "
            f"lr={lr_str}, time={elapsed:.2f}s"
        )
        log(msg)


class BatchLogger(keras.callbacks.Callback):
    """Her N batch'te loss/acc ve throughput loglar."""

    def __init__(self, log_every: int):
        super().__init__()
        self.log_every = max(1, int(log_every))
        self._train_start_time: Optional[float] = None

    def on_train_begin(self, logs=None):
        self._train_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if LOG_EVERY_BATCHES <= 0:
            return
        logs = logs or {}
        batch_idx = batch + 1
        if batch_idx % self.log_every != 0:
            return
        elapsed = time.time() - (self._train_start_time or time.time())
        batch_size = self.params.get("batch_size", 1)
        samples_seen = batch_idx * batch_size
        throughput = samples_seen / elapsed if elapsed > 0 else 0.0
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        log(
            f"[Batch {batch_idx}] loss={loss:.4f} acc={acc:.4f} "
            f"throughput={throughput:.1f} img/s"
        )


def train_model(
    model,
    train_gen,
    val_gen,
    output_dir: str,
    model_name: str = "",
    class_weight: Optional[Dict[int, float]] = None,
):
    """Train the model and save training curves. class_weight improves minority classes (e.g. partly_cloudy)."""
    callbacks = []
    if HAS_TENSORFLOW:
        # Eski global training_log.csv'yi koru
        csv_logger_global = keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"), append=False
        )
        # Modele özel CSV
        if model_name:
            csv_logger_model = keras.callbacks.CSVLogger(
                os.path.join(output_dir, f"training_log_{model_name}.csv"),
                append=False,
            )
        else:
            csv_logger_model = None

        callbacks = [csv_logger_global]
        if csv_logger_model is not None:
            callbacks.append(csv_logger_model)
        callbacks.append(EpochLogger())
        if LOG_EVERY_BATCHES > 0:
            callbacks.append(BatchLogger(LOG_EVERY_BATCHES))

        # Erken durdurma: val_loss 20 epoch iyileşmezse dur, en iyi ağırlıkları geri yükle
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)
        # Öğrenme oranı: val_loss 7 epoch düşmezse LR'yi yarıya indir (daha iyi metrik)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        )
        callbacks.append(reduce_lr)

    fit_kwargs = dict(
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1,
        callbacks=callbacks,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(train_gen, **fit_kwargs)

    hist = history.history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(hist["accuracy"], label="train_acc")
    plt.plot(hist["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    plt.savefig(os.path.join(output_dir, f"training_curves{suffix}.png"))
    plt.close()

    return model, history


def evaluate_model(
    model,
    test_gen,
    label_to_idx: Dict[str, int],
    output_dir: str,
) -> Dict:
    """Evaluate model on test set and compute metrics & plots."""
    probs = model.predict(test_gen, verbose=1)
    y_pred_idx = probs.argmax(axis=1)

    y_true_idx = test_gen.classes

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    y_true_labels = [idx_to_label[i] for i in y_true_idx]
    y_pred_labels = [idx_to_label[i] for i in y_pred_idx]

    acc = accuracy_score(y_true_labels, y_pred_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average="weighted", zero_division=0
    )

    metrics_dict = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

    # Ek metrikler: balanced accuracy, MCC
    try:
        bal_acc = balanced_accuracy_score(y_true_labels, y_pred_labels)
        metrics_dict["balanced_accuracy"] = float(bal_acc)
    except Exception:
        metrics_dict["balanced_accuracy"] = None
    try:
        mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
        metrics_dict["mcc"] = float(mcc)
    except Exception:
        metrics_dict["mcc"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(idx_to_label.values()))
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(idx_to_label.values()),
            yticklabels=list(idx_to_label.values()),
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(
            ticks=np.arange(len(idx_to_label)),
            labels=list(idx_to_label.values()),
            rotation=45,
        )
        plt.yticks(
            ticks=np.arange(len(idx_to_label)),
            labels=list(idx_to_label.values()),
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    name_tag = getattr(model, "name", "")
    if name_tag:
        cm_suffix = f"_{name_tag}"
    else:
        cm_suffix = ""
    plt.savefig(os.path.join(output_dir, f"confusion_matrix{cm_suffix}.png"))
    plt.close()

    # ROC-AUC (multi-class OVR) if possible
    try:
        num_classes = len(idx_to_label)
        y_true_onehot = tf.keras.utils.to_categorical(y_true_idx, num_classes=num_classes)
        roc_auc = roc_auc_score(y_true_onehot, probs, multi_class="ovr")
        metrics_dict["roc_auc_ovr"] = float(roc_auc)
    except Exception:
        metrics_dict["roc_auc_ovr"] = None

    # ROC Curves
    try:
        num_classes = probs.shape[1]
        plt.figure(figsize=(7, 6))
        for i in range(num_classes):
            y_true_bin = (y_true_idx == i).astype(int)
            y_score = probs[:, i]
            if len(np.unique(y_true_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc_i = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{idx_to_label[i]} (AUC={roc_auc_i:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Eğrileri (OVR)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"roc_curves{cm_suffix}.png"))
        plt.close()
    except Exception:
        pass

    metrics_path = os.path.join(output_dir, f"model_metrics{cm_suffix}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # Konsola metrik dökümü
    log("=== Test Metrikleri ===")
    for k, v in metrics_dict.items():
        log(f"{k}: {v}")

    # Classification report (sınıf bazlı)
    try:
        report = classification_report(
            y_true_labels,
            y_pred_labels,
            labels=list(idx_to_label.values()),
            zero_division=0,
        )
        log("=== Classification Report ===")
        for line in report.splitlines():
            log(line)
    except Exception as e:
        log(f"classification_report hesaplanamadı: {e}", logging.WARNING)

    # Confusion matrix sayısal tablo
    log("=== Confusion Matrix (sayısal) ===")
    labels = list(idx_to_label.values())
    header = "true\\pred".ljust(15) + " " + " ".join(lbl.rjust(15) for lbl in labels)
    log(header)
    for i, true_lbl in enumerate(labels):
        row_vals = " ".join(str(cm[i, j]).rjust(15) for j in range(len(labels)))
        log(true_lbl.ljust(15) + " " + row_vals)

    # Hiç tahmin edilmeyen sınıflar için uyarı
    missing_classes = []
    for lbl in labels:
        if lbl not in y_pred_labels:
            missing_classes.append(lbl)
    if missing_classes:
        log(
            "UYARI: Aşağıdaki sınıflar hiç tahmin edilmedi: "
            + ", ".join(missing_classes),
            logging.WARNING,
        )
        log(
            "Öneri: optimize-metrics modu ile class_weight/fine-tune/augmentasyonu artırmayı düşünebilirsiniz.",
            logging.INFO,
        )

    return metrics_dict


def plot_cloudiness_vs_labels(
    cloud_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Sınıf (folder veya category) bazında bulutluluk dağılımı; seaborn y kolon adı ile."""
    df = cloud_df.dropna(subset=["cloudiness"]).copy()
    label_col = "folder" if "folder" in df.columns else "category"
    if df.empty or label_col not in df.columns:
        return
    df["cloudiness_pct"] = df["cloudiness"] * 100

    if MONTH_FILTER and "datetime" in df and df["datetime"].notnull().any():
        df = df[df["datetime"].dt.to_period("M").astype(str) == MONTH_FILTER]
    df = df[~df["is_night"]].copy()
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.boxplot(data=df, x=label_col, y="cloudiness_pct")
        plt.ylabel("Bulutluluk (%)")
        plt.xlabel("Label")
        plt.title("Sınıf Bazında Bulutluluk Dağılımı (Boxplot)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cloudiness_by_label_boxplot.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x=label_col, y="cloudiness_pct", cut=0)
        plt.ylabel("Bulutluluk (%)")
        plt.xlabel("Label")
        plt.title("Sınıf Bazında Bulutluluk Dağılımı (Violin)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cloudiness_by_label_violin.png"))
        plt.close()
    else:
        labels = sorted(df[label_col].unique())
        data = [df[df[label_col] == lbl]["cloudiness_pct"].values for lbl in labels]
        plt.boxplot(data, labels=labels)
        plt.ylabel("Bulutluluk (%)")
        plt.xlabel("Label")
        plt.title("Sınıf Bazında Bulutluluk Dağılımı (Boxplot)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cloudiness_by_label_boxplot.png"))
        plt.close()


def list_outputs(output_dir: str) -> None:
    """List contents of outputs directory."""
    print("\n=== outputs/ içeriği ===")
    if not os.path.isdir(output_dir):
        print("outputs/ klasörü bulunamadı.")
        return
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


def diagnose_hardware(force_cpu: bool, output_dir: str) -> None:
    """Log TF/CPU/GPU durumu, nvidia-smi çıktısı ve hardware_info.json kaydı."""
    info: Dict[str, object] = {}

    if not HAS_TENSORFLOW:
        log("TensorFlow bulunamadı; yalnızca CPU ile devam edilecek.", logging.WARNING)
        info["tensorflow_available"] = False
    else:
        info["tensorflow_available"] = True
        info["tf_version"] = tf.__version__
        try:
            info["tf_built_with_cuda"] = tf.test.is_built_with_cuda()
        except Exception:
            info["tf_built_with_cuda"] = None
        try:
            build_info = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        except Exception:
            build_info = {}
        info["tf_build_info"] = build_info
        info["devices"] = {
            "CPU": [str(d) for d in tf.config.list_physical_devices("CPU")],
            "GPU": [str(d) for d in tf.config.list_physical_devices("GPU")],
        }

    # nvidia-smi teşhisi
    nvidia: Dict[str, object] = {}
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        nvidia["raw"] = result.stdout
        if result.returncode == 0:
            log("nvidia-smi çıktısı alındı.")
        else:
            log(f"nvidia-smi hata kodu: {result.returncode}", logging.WARNING)
    except FileNotFoundError:
        log("nvidia-smi bulunamadı; NVIDIA GPU sürücüsü PATH'te olmayabilir.", logging.WARNING)
        nvidia["raw"] = None
    except Exception as e:
        log(f"nvidia-smi çalıştırılırken hata: {e}", logging.WARNING)
        nvidia["raw"] = None

    # Detaylı GPU query (varsa)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        nvidia["query"] = result.stdout.strip().splitlines()
    except Exception:
        nvidia["query"] = None

    info["nvidia_smi"] = nvidia

    # GPU kullanımı / force_cpu
    if force_cpu:
        log("force_cpu=True: GPU olsa bile CPU kullanılacak.")
    elif HAS_TENSORFLOW:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                log(f"{len(gpus)} GPU bulundu, memory_growth aktif. GPU AKTİF.")
            except Exception as e:
                log(f"GPU memory_growth ayarlanamadı: {e}", logging.WARNING)
        else:
            log("GPU bulunamadı; CPU ile devam edilecek.", logging.WARNING)
            log(
                "GPU için kontrol listesi:\n"
                "  - nvidia-smi çıktısı var mı?\n"
                "  - Doğru Python interpreter mı (py -3.11 ile aynı mı)?\n"
                "  - TF build CUDA ile derlenmiş mi?\n"
                "  - Native Windows TF 2.11+ için GPU desteği sınırlı; WSL2 + tensorflow[and-cuda] önerilir.",
                logging.INFO,
            )

    # JSON'a yaz
    try:
        ensure_output_dir(output_dir)
        hw_path = os.path.join(output_dir, "hardware_info.json")
        with open(hw_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        log(f"Donanım bilgisi kaydedildi: {hw_path}")
    except Exception as e:
        log(f"Donanım bilgisi JSON yazılamadı: {e}", logging.WARNING)


def main() -> None:
    """Main pipeline: dataset profiling, cloudiness, ML model, and reporting."""
    global MONTH_FILTER
    set_global_seeds(SEED)
    ensure_output_dir(OUTPUT_DIR)

    # 1) Sistem bilgisi ve donanım teşhisi (logging henüz kurulmuş olmalı)
    print_system_info(OUTPUT_DIR)
    diagnose_hardware(FORCE_CPU, OUTPUT_DIR)

    # 2) Veri klasörünü çöz
    with timed_step("Veri klasörü bulundu"):
        try:
            data_dir = resolve_data_dir(DATA_DIR)
        except RuntimeError as e:
            log(str(e), logging.ERROR)
            sys.exit(1)
        log(f"Veri klasörü: {data_dir}")

    # 3) Veri seti keşfi
    with timed_step("Veri taraması & dataset profil"):
        df, corrupted = discover_images(data_dir)
        if df.empty:
            log("Hiç geçerli görüntü bulunamadı. Çıkılıyor.", logging.ERROR)
            sys.exit(1)
        df_profile = profile_dataset(df, corrupted)

    # MONTH_FILTER None ve AUTO_MONTH_LATEST ise en yeni ayı seç (rapor/grafikler bu aya göre)
    if MONTH_FILTER is None and AUTO_MONTH_LATEST and "month" in df_profile.columns:
        months = df_profile["month"].dropna().unique().tolist()
        if months:
            MONTH_FILTER = sorted(months)[-1]
            print(f"AUTO_MONTH_LATEST: MONTH_FILTER = {MONTH_FILTER}")

        # MONTH_FILTER None ve AUTO_MONTH_LATEST ise en yeni ayı seç (rapor/grafikler bu aya göre)
        if MONTH_FILTER is None and AUTO_MONTH_LATEST and "month" in df_profile.columns:
            months = df_profile["month"].dropna().unique().tolist()
            if months:
                MONTH_FILTER = sorted(months)[-1]
                log(f"AUTO_MONTH_LATEST: MONTH_FILTER = {MONTH_FILTER}")

        # Bozuk dosyalar listesi kaydet
        broken_path = os.path.join(OUTPUT_DIR, "broken_files.txt")
        with open(broken_path, "w", encoding="utf-8") as f:
            for p in corrupted:
                f.write(p + "\n")
        log(f"Bozuk dosya listesi kaydedildi: {broken_path}")

        # Dataset profile csv
        profile_csv_path = os.path.join(OUTPUT_DIR, "dataset_profile.csv")
        df_profile.to_csv(profile_csv_path, index=False)
        log(f"Dataset profil CSV kaydedildi: {profile_csv_path}")

    # 4) Tüm görüntüler için bulutluluk hesapla + debug (ROI 0/1, sky_pixels==0)
    with timed_step("Bulutluluk hesaplama"):
        debug_info = {}
        cloud_df = compute_cloudiness_for_all(df_profile, debug_info=debug_info)
        with open(os.path.join(OUTPUT_DIR, "roi_debug.txt"), "w", encoding="utf-8") as f:
            f.write(f"ROI mask unique değerleri (beklenen: {{0, 1}}): {sorted(debug_info.get('roi_unique_values', set()))}\n")
            f.write(f"sky_pixels==0 olan dosya sayısı: {len(debug_info.get('sky_pixels_zero_paths', []))}\n")
            for p in debug_info.get("sky_pixels_zero_paths", []):
                f.write(p + "\n")
        log(f"Debug: roi_debug.txt yazıldı (ROI unique={sorted(debug_info.get('roi_unique_values', set()))}, sky_pixels=0: {len(debug_info.get('sky_pixels_zero_paths', []))})")

        # per_image_cloudiness.csv
        per_image_csv = os.path.join(OUTPUT_DIR, "per_image_cloudiness.csv")
        cloud_df.to_csv(per_image_csv, index=False)
        log(f"Per-image bulutluluk CSV kaydedildi: {per_image_csv}")

    # 5) Özet ve rapor görselleri
    with timed_step("Zaman serisi ve özet grafikleri"):
        summarize_cloudiness(cloud_df)
        plot_time_series_and_histograms(cloud_df, OUTPUT_DIR)
        save_segmentation_examples(cloud_df, OUTPUT_DIR, num_examples=4)
        plot_cloudiness_vs_labels(cloud_df, OUTPUT_DIR)

    # 5) ML modelleri: TRAIN_ML=True veya --train-ml ile çalıştırıldıysa
    if not TRAIN_ML or not HAS_TENSORFLOW:
        if not HAS_TENSORFLOW:
            print("\nTensorFlow yüklü değil. ML modeli atlanıyor.")
        else:
            print("\nTRAIN_ML=False; ML eğitimi atlandı. Açmak için TRAIN_ML=True veya --train-ml kullanın.")
        list_outputs(OUTPUT_DIR)
        return

    try:
        train_df, val_df, test_df, label_to_idx = prepare_splits(cloud_df)
    except RuntimeError as e:
        log(f"Model eğitimi atlandı: {e}")
        list_outputs(OUTPUT_DIR)
        return

    log_step("ML Split & Generator")
    log(
        f"Train/Val/Test örnek sayıları: "
        f"{len(train_df)} / {len(val_df)} / {len(test_df)}"
    )
    label_col = ML_LABEL_COLUMN if ML_LABEL_COLUMN in train_df.columns else "folder"
    log(f"ML label_col: {label_col}")
    log("Sınıf dağılımı (train_df):")
    log(str(train_df[label_col].value_counts()))
    log(f"label_to_idx: {label_to_idx}")

    # Sınıf dengesizliğini azaltmak için class_weight (az örnekli sınıflar daha yüksek ağırlık)
    y_train = train_df[label_col].values
    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(
        "balanced", classes=classes, y=y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}
    # Keras için sınıf indeksi (label_to_idx değeri) -> ağırlık
    label_to_weight = {label: float(class_weights_arr[list(classes).index(label)]) for label in classes}
    keras_class_weight = {label_to_idx[lbl]: w for lbl, w in label_to_weight.items()}
    log("Class weights (balanced, metrik iyileştirmesi için):")
    for lbl in sorted(label_to_idx.keys()):
        log(f"  {lbl} (idx={label_to_idx[lbl]}): {keras_class_weight[label_to_idx[lbl]]:.4f}")

    train_gen, val_gen, test_gen = build_generators(
        train_df, val_df, test_df, label_to_idx
    )

    # Sınıf isimleri çıktı indeks sırasına göre (backend tek dosyadan anlasın)
    num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names_ordered = [idx_to_label[i] for i in range(num_classes)]

    # İki backbone: önce B0 sonra B3; en iyi model seçilip best_cloud_classifier.keras + config yazılacak
    best_candidate = None  # (backbone_name, metrics_dict, model_path)
    for backbone_name in ["EfficientNetB0", "EfficientNetB3"]:
        log_step(f"ML Eğitimi – {backbone_name}")
        model = build_model(num_classes=num_classes, backbone_name=backbone_name)

        # Model özetini hem konsola yaz hem dosyaya kaydet
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        summary_text = buf.getvalue()
        log(summary_text)
        summary_path = os.path.join(OUTPUT_DIR, f"model_summary_{backbone_name}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        log(f"Model özeti kaydedildi: {summary_path}")

        model, history = train_model(
            model, train_gen, val_gen, OUTPUT_DIR, model_name=backbone_name,
            class_weight=keras_class_weight,
        )

        # Değerlendirme
        metrics_dict = evaluate_model(model, test_gen, label_to_idx, OUTPUT_DIR)
        log("\n=== Model Metrikleri – {name} ===".format(name=backbone_name))
        for k, v in metrics_dict.items():
            log(f"{k}: {v}")

        # Her backbone için ayrı dosya kaydet
        model_path = os.path.join(
            OUTPUT_DIR, f"cloud_classifier_{backbone_name}.keras"
        )
        model.save(model_path)
        log(f"Model kaydedildi: {model_path}")

        # En iyi model adayı: önce f1_macro, yoksa accuracy
        f1 = metrics_dict.get("f1_macro") or 0.0
        acc = metrics_dict.get("accuracy") or 0.0
        if best_candidate is None or (f1 > (best_candidate[1].get("f1_macro") or 0.0)) or (
            f1 == (best_candidate[1].get("f1_macro") or 0.0) and acc > (best_candidate[1].get("accuracy") or 0.0)
        ):
            best_candidate = (backbone_name, metrics_dict, model_path)

    # Best model: tek dosya + config (backend/web entegrasyonu için)
    if best_candidate is not None:
        best_backbone, best_metrics, best_src_path = best_candidate
        best_model_path = os.path.join(OUTPUT_DIR, "best_cloud_classifier.keras")
        shutil.copy2(best_src_path, best_model_path)
        log(f"Best model kopyalandı: {best_backbone} -> {best_model_path}")

        config = {
            "model_file": "best_cloud_classifier.keras",
            "backbone": best_backbone,
            "class_names": class_names_ordered,
            "num_classes": num_classes,
            "input_shape": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
            "preprocessing": "efficientnet",
            "preprocessing_note": "Apply keras.applications.efficientnet.preprocess_input to RGB image (uint8 0-255) before predict.",
            "test_metrics": {k: v for k, v in best_metrics.items() if v is not None},
            "inference_example": (
                "import json\n"
                "from tensorflow import keras\n"
                "import keras.applications.efficientnet as eff\n"
                "with open('best_model_config.json') as f:\n"
                "    cfg = json.load(f)\n"
                "model = keras.models.load_model(cfg['model_file'])\n"
                "x = eff.preprocess_input(your_rgb_image)  # shape (H,W,3), then batch\n"
                "pred = model.predict(x)\n"
                "label = cfg['class_names'][pred[0].argmax()]"
            ),
        }
        config_path = os.path.join(OUTPUT_DIR, "best_model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        log(f"Best model config kaydedildi: {config_path}")

    # 8) outputs/ içeriğini listele
    list_outputs(OUTPUT_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gökyüzü bulutluluk analizi ve raporlama")
    parser.add_argument("--train-ml", action="store_true", help="ML model eğitimini çalıştır (TRAIN_ML=True)")
    parser.add_argument("--log-file", type=str, default=None, help="Log dosyası (ör. outputs/run.log)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="GPU olsa bile CPU kullan (GPU memory_growth uygulanmaz)",
    )
    parser.add_argument(
        "--print-system-info-only",
        action="store_true",
        help="Sadece sistem/donanım bilgisini yazdır ve çık",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="İlerleme barlarını/devam loglarını azalt (gelecek fazlar için)",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        help="N>0 ise her N batch'te loss/acc + throughput loglar",
    )
    args = parser.parse_args()

    # Logging kurulumunu argümanlarla yap
    ensure_output_dir(OUTPUT_DIR)
    setup_logging(args.log_level, args.log_file)

    if args.train_ml:
        globals()["TRAIN_ML"] = True
    globals()["NO_PROGRESS"] = bool(args.no_progress)
    globals()["FORCE_CPU"] = bool(args.force_cpu)
    globals()["LOG_EVERY_BATCHES"] = max(0, int(args.log_every_batches))

    if args.print_system_info_only:
        print_system_info(OUTPUT_DIR)
        if HAS_TENSORFLOW:
            diagnose_hardware(FORCE_CPU, OUTPUT_DIR)
        sys.exit(0)

    main()
