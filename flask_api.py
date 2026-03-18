"""
FLASK API SERVER FOR MULTIPLE DISEASE PREDICTIONS
Hỗ trợ dự đoán 4 loại bệnh: Tim, Tiểu đường, Huyết áp cao, Đột quỵ
Hỗ trợ thêm dự đoán bằng ảnh: Sỏi thận, Viêm phổi
Endpoint tích hợp với HealthManagement C# Application
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import json
import base64
import builtins
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont

try:
    import torch
    try:
        from image_cnn_utils import load_binary_cnn_model, predict_probability_and_gradcam
    except Exception:
        from AIModel.image_cnn_utils import load_binary_cnn_model, predict_probability_and_gradcam
    TORCH_IMAGE_SUPPORT = True
except Exception as torch_image_error:
    torch = None
    load_binary_cnn_model = None
    predict_probability_and_gradcam = None
    TORCH_IMAGE_SUPPORT = False
    print(f"⚠️  PyTorch image pipeline unavailable: {torch_image_error}")

def safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        normalized_args = [str(arg).encode('ascii', errors='ignore').decode('ascii') for arg in args]
        builtins.print(*normalized_args, **kwargs)

print = safe_print

app = Flask(__name__)
CORS(app)  # Enable CORS for C# application

# ============================================================
# LOAD TRAINED MODELS
# ============================================================
print("Loading trained models...")

MODELS = {}
SCALERS = {}
FEATURE_SETS = {}
THRESHOLD_META = {}
DISEASE_TYPES = ['heart_disease', 'diabetes', 'hypertension', 'stroke']

IMAGE_MODELS = {}
IMAGE_LABELS = {}
IMAGE_META = {}
IMAGE_DISEASE_TYPES = ['kidney_stone_image', 'pneumonia_image']

for disease in DISEASE_TYPES:
    try:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(model_dir, f'{disease}_model.pkl')
        scaler_file = os.path.join(model_dir, f'{disease}_scaler.pkl')
        features_file = os.path.join(model_dir, f'{disease}_features.pkl')
        threshold_meta_file = os.path.join(model_dir, f'{disease}_threshold_meta.pkl')

        if os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(features_file):
            MODELS[disease] = joblib.load(model_file)
            SCALERS[disease] = joblib.load(scaler_file)
            FEATURE_SETS[disease] = joblib.load(features_file)
            if os.path.exists(threshold_meta_file):
                try:
                    loaded_meta = joblib.load(threshold_meta_file)
                    if isinstance(loaded_meta, dict):
                        THRESHOLD_META[disease] = loaded_meta
                except Exception as meta_err:
                    print(f"⚠️  Cannot load threshold meta for {disease}: {meta_err}")
            print(f"✅ {disease.upper()} model loaded!")
        else:
            print(f"⚠️  {disease.upper()} model files not found - training needed")
    except Exception as e:
        print(f"⚠️  Error loading {disease}: {e}")

if not MODELS:
    print("❌ No models loaded! Please run training scripts first.")
    print("   python train_heart_disease.py")
    print("   python train_diabetes.py")
    print("   python train_hypertension.py")
    print("   python train_stroke.py")

print("Loading image models...")
for disease in IMAGE_DISEASE_TYPES:
    try:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_pt_file = os.path.join(model_dir, f'{disease}_model.pt')
        model_pkl_file = os.path.join(model_dir, f'{disease}_model.pkl')
        labels_file = os.path.join(model_dir, f'{disease}_labels.pkl')
        meta_file = os.path.join(model_dir, f'{disease}_meta.json')

        meta = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        IMAGE_META[disease] = meta

        if os.path.exists(model_pt_file) and TORCH_IMAGE_SUPPORT:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, target_layer_name = load_binary_cnn_model(model_pt_file, meta, device=device)
            IMAGE_MODELS[disease] = {
                'type': 'pytorch_cnn',
                'model': model,
                'device': device,
                'target_layer': target_layer_name,
            }
            if os.path.exists(labels_file):
                IMAGE_LABELS[disease] = joblib.load(labels_file)
            print(f"✅ {disease.upper()} PyTorch image model loaded!")
        elif os.path.exists(model_pkl_file):
            IMAGE_MODELS[disease] = joblib.load(model_pkl_file)
            if os.path.exists(labels_file):
                IMAGE_LABELS[disease] = joblib.load(labels_file)
            print(f"✅ {disease.upper()} sklearn image model loaded!")
        else:
            print(f"⚠️  {disease.upper()} image model not found - training needed")
    except Exception as e:
        print(f"⚠️  Error loading image model {disease}: {e}")

# ============================================================
# DISEASE DESCRIPTIONS
# ============================================================
DISEASE_INFO = {
    'heart_disease': {
        'name': 'Bệnh Tim (Heart Disease)',
        'description': 'Dự đoán nguy cơ bệnh tim dựa trên các chỉ số lâm sàng',
        'features': ['age', 'sex', 'chestpaintype', 'restingbp', 'cholesterol', 
                    'fastingbs', 'maxhr', 'exerciseangina']
    },
    'diabetes': {
        'name': 'Bệnh Tiểu Đường (Diabetes)',
        'description': 'Dự đoán nguy cơ bệnh tiểu đường',
        'features': ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
                    'insulin', 'bmi', 'diabetespedigreefunction', 'age']
    },
    'hypertension': {
        'name': 'Huyết Áp Cao (Hypertension)',
        'description': 'Dự đoán nguy cơ tăng huyết áp',
        'features': ['age', 'gender', 'systolicbp', 'diastolicbp', 'cholesterol',
                    'bmi', 'smoking', 'alcohol', 'physicalactivity']
    },
    'stroke': {
        'name': 'Đột Quỵ (Stroke)',
        'description': 'Dự đoán nguy cơ đột quỵ',
        'features': ['age', 'gender', 'hypertension', 'heartdisease', 'smoking',
                    'bmi', 'glucose']
    },
    'kidney_stone_image': {
        'name': 'Sỏi Thận (Kidney Stone) - Ảnh CT',
        'description': 'Dự đoán nguy cơ sỏi thận từ ảnh CT',
        'features': ['image_file']
    },
    'pneumonia_image': {
        'name': 'Viêm Phổi (Pneumonia) - Ảnh X-Quang',
        'description': 'Dự đoán nguy cơ viêm phổi từ ảnh X-ray ngực',
        'features': ['image_file']
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def convert_sex_to_numeric(sex_value):
    """Convert sex from string to numeric"""
    if isinstance(sex_value, str):
        sex_value = sex_value.strip().lower()
        if sex_value in ['male', 'm', 'nam']:
            return 1
        elif sex_value in ['female', 'f', 'nữ', 'nu']:
            return 0
    return int(sex_value) if sex_value else 0

def normalize_key(key):
    return ''.join(ch for ch in str(key).strip().lower() if ch.isalnum())

def to_numeric_value(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return 0.0
        return float(value)

    text = str(value).strip().lower()
    if text in ['male', 'm', 'nam']:
        return 1.0
    if text in ['female', 'f', 'nữ', 'nu']:
        return 0.0
    if text in ['yes', 'true', 'co']:
        return 1.0
    if text in ['no', 'false', 'khong', 'không']:
        return 0.0

    try:
        return float(text)
    except Exception:
        return 0.0

FEATURE_ALIASES = {
    'heart_disease': {
        'chestpaintype': 'cp',
        'restingbp': 'trestbps',
        'cholesterol': 'chol',
        'fastingbs': 'fbs',
        'maxhr': 'thalach',
        'exerciseangina': 'exang'
    },
    'stroke': {
        'avgbloodpressure': 'Glucose',
        'avg_glucose_level': 'Glucose'
    }
}

def build_input_dataframe(disease_type, disease_data, feature_names, scaler):
    if not isinstance(disease_data, dict):
        disease_data = {}

    normalized_input = {
        normalize_key(k): v for k, v in disease_data.items()
    }

    alias_map = FEATURE_ALIASES.get(disease_type, {})
    alias_pairs = [
        (normalize_key(src), normalize_key(dst)) for src, dst in alias_map.items()
    ]

    scaler_means = getattr(scaler, 'mean_', None)
    row = {}

    for index, feature_name in enumerate(feature_names):
        feature_norm = normalize_key(feature_name)
        value = None

        if feature_norm in normalized_input:
            value = normalized_input[feature_norm]
        else:
            for src_norm, dst_norm in alias_pairs:
                if dst_norm == feature_norm and src_norm in normalized_input:
                    value = normalized_input[src_norm]
                    break

        if value is None:
            if scaler_means is not None and len(scaler_means) > index:
                value = scaler_means[index]
            else:
                value = 0

        row[feature_name] = to_numeric_value(value)

    return pd.DataFrame([row], columns=feature_names)

def get_risk_level(probability, disease_type):
    """Xác định mức độ nguy cơ và lời khuyên dựa trên loại bệnh"""
    risk_percentage = probability * 100
    disease_name = DISEASE_INFO.get(disease_type, {}).get('name', 'bệnh')
    
    if probability >= 0.7:
        result = "Nguy cơ cao"
        recommendation = f"⚠️ Khuyến nghị gặp bác sĩ khám chuyên khoa ngay để kiểm tra {disease_name}."
        details = f"Model AI dự đoán nguy cơ {disease_name} cao ({risk_percentage:.1f}%). Cần theo dõi sát sao."
    elif probability >= 0.4:
        result = "Nguy cơ trung bình"
        recommendation = f"Nên cải thiện lối sống: ăn uống lành mạnh, tập thể dục đều đặn, giảm stress."
        details = f"Model AI phát hiện nguy cơ {disease_name} trung bình ({risk_percentage:.1f}%). Theo dõi định kỳ."
    else:
        result = "Nguy cơ thấp"
        recommendation = f"Tiếp tục duy trì lối sống lành mạnh. Khám sức khỏe định kỳ."
        details = f"Model AI đánh giá nguy cơ {disease_name} thấp ({risk_percentage:.1f}%). Tốt!"
    
    return result, risk_percentage, recommendation, details


def get_decision_threshold(disease_type):
    meta = THRESHOLD_META.get(disease_type, {})
    threshold = meta.get('threshold', 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5
    if threshold <= 0 or threshold >= 1:
        threshold = 0.5
    return threshold


def get_image_decision_threshold(disease_type):
    meta = IMAGE_META.get(disease_type, {})
    threshold = meta.get('threshold', 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    if threshold <= 0 or threshold >= 1:
        threshold = 0.5
    return threshold


def extract_image_features(image_bytes, disease_type):
    meta = IMAGE_META.get(disease_type, {})
    feature_size = meta.get('sklearn_feature_size', [96, 96])
    try:
        width = int(feature_size[0])
        height = int(feature_size[1])
    except Exception:
        width, height = 96, 96

    img = Image.open(BytesIO(image_bytes)).convert('L')
    img_small = img.resize((width, height))
    arr = np.asarray(img_small, dtype=np.float32) / 255.0

    flat = arr.flatten()
    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=True)
    stats = np.array([
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
    ], dtype=np.float32)

    feats = np.concatenate([flat, hist.astype(np.float32), stats]).astype(np.float32)
    return feats.reshape(1, -1)


def get_positive_class_index(disease_type, labels):
    if not labels:
        return 1

    labels_lower = [str(lbl).strip().lower() for lbl in labels]
    if disease_type == 'kidney_stone_image':
        for idx, lbl in enumerate(labels_lower):
            if lbl in ['stone', 'kidney_stone', 'sỏi thận', 'soi than']:
                return idx
        return 1 if len(labels_lower) > 1 else 0

    if disease_type == 'pneumonia_image':
        for idx, lbl in enumerate(labels_lower):
            if lbl in ['pneumonia', 'viêm phổi', 'viem phoi']:
                return idx
        return 1 if len(labels_lower) > 1 else 0

    return 1 if len(labels_lower) > 1 else 0


def resize_image_for_visualization(img, max_side=640):
    width, height = img.size
    if max(width, height) <= max_side:
        return img.copy()

    scale = max_side / float(max(width, height))
    resized = (
        max(1, int(width * scale)),
        max(1, int(height * scale))
    )
    return img.resize(resized)


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def resize_attention_map(attention_map, width, height):
    arr = np.asarray(attention_map, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr -= arr.min()
    arr /= max(float(arr.max()), 1e-6)

    if arr.shape != (height, width):
        resized = Image.fromarray((arr * 255.0).astype(np.uint8)).resize((width, height), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0

    return np.clip(arr, 0.0, 1.0)


def colorize_attention_map(attention_map, reference_mode):
    base = np.clip(np.asarray(attention_map, dtype=np.float32), 0.0, 1.0)
    if reference_mode:
        red = base * 0.12
        green = np.clip((base * 0.72) + 0.05, 0.0, 1.0)
        blue = np.clip(base * 1.05, 0.0, 1.0)
    else:
        red = np.clip(base * 1.35, 0.0, 1.0)
        green = np.clip(np.power(base, 0.72), 0.0, 1.0)
        blue = base * 0.10
    return np.dstack([red, green, blue])


def weighted_bounds(weights, lower_q=0.03, upper_q=0.97):
    arr = np.clip(np.asarray(weights, dtype=np.float32), 0.0, None)
    if arr.size <= 1:
        return 0, int(arr.size)

    total = float(arr.sum())
    if total <= 1e-6:
        return 0, int(arr.size)

    cdf = np.cumsum(arr)
    left = int(np.searchsorted(cdf, total * float(lower_q), side='left'))
    right = int(np.searchsorted(cdf, total * float(upper_q), side='left')) + 1
    left = max(0, min(left, arr.size - 1))
    right = max(left + 1, min(right, arr.size))
    return left, right


def enforce_min_span(start, end, min_span, limit):
    span = int(end - start)
    min_span = max(1, int(min_span))
    limit = max(1, int(limit))
    if span >= min_span:
        return max(0, int(start)), min(limit, int(end))

    center = int((start + end) // 2)
    new_start = max(0, center - (min_span // 2))
    new_end = min(limit, new_start + min_span)
    new_start = max(0, new_end - min_span)
    return int(new_start), int(new_end)


def normalize_bbox(bbox, width, height):
    if bbox is None:
        return 0, 0, int(width), int(height)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def estimate_anatomy_bbox(gray_arr, disease_type=None):
    arr = np.asarray(gray_arr, dtype=np.float32)
    height, width = arr.shape
    if height < 8 or width < 8:
        return 0, 0, width, height

    disease_type = str(disease_type or '').strip().lower()
    blurred = np.asarray(
        Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)),
        dtype=np.float32
    )

    base_percentile = 12.0 if disease_type == 'pneumonia_image' else 18.0
    baseline = float(np.percentile(blurred, base_percentile))
    signal = np.clip(blurred - baseline, 0.0, None)

    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    sigma_x = max(width * (0.34 if disease_type == 'pneumonia_image' else 0.38), 1.0)
    sigma_y = max(height * (0.42 if disease_type == 'pneumonia_image' else 0.40), 1.0)
    center_prior = np.exp(
        -(
            ((xx - cx) ** 2) / (2.0 * sigma_x * sigma_x) +
            ((yy - cy) ** 2) / (2.0 * sigma_y * sigma_y)
        )
    )

    weighted = signal * center_prior
    if float(weighted.sum()) <= 1e-6:
        return 0, 0, width, height

    x1, x2 = weighted_bounds(weighted.sum(axis=0), lower_q=0.02, upper_q=0.98)
    y1, y2 = weighted_bounds(weighted.sum(axis=1), lower_q=0.02, upper_q=0.98)

    pad_x = int((x2 - x1) * (0.10 if disease_type == 'pneumonia_image' else 0.12))
    pad_y = int((y2 - y1) * (0.08 if disease_type == 'pneumonia_image' else 0.10))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    min_w = max(24, int(width * (0.42 if disease_type == 'pneumonia_image' else 0.32)))
    min_h = max(24, int(height * (0.56 if disease_type == 'pneumonia_image' else 0.36)))
    x1, x2 = enforce_min_span(x1, x2, min_w, width)
    y1, y2 = enforce_min_span(y1, y2, min_h, height)

    return normalize_bbox((x1, y1, x2, y2), width, height)


def build_soft_bbox_mask(width, height, bbox, blur_ratio=0.045):
    x1, y1, x2, y2 = normalize_bbox(bbox, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    blur_radius = max(2, int(min(width, height) * float(blur_ratio)))
    softened = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.asarray(softened, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def build_pneumonia_lung_mask(width, height, anatomy_bbox=None):
    x1, y1, x2, y2 = normalize_bbox(anatomy_bbox, width, height)
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))

    yy, xx = np.mgrid[0:height, 0:width]
    left_lung = (
        ((xx - (x1 + (bbox_w * 0.34))) / max(bbox_w * 0.18, 1.0)) ** 2 +
        ((yy - (y1 + (bbox_h * 0.54))) / max(bbox_h * 0.34, 1.0)) ** 2
    ) <= 1.0
    right_lung = (
        ((xx - (x1 + (bbox_w * 0.66))) / max(bbox_w * 0.18, 1.0)) ** 2 +
        ((yy - (y1 + (bbox_h * 0.54))) / max(bbox_h * 0.34, 1.0)) ** 2
    ) <= 1.0
    return (left_lung | right_lung).astype(np.float32)


def build_kidney_focus_mask(width, height, anatomy_bbox=None):
    x1, y1, x2, y2 = normalize_bbox(anatomy_bbox, width, height)
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))

    yy, xx = np.mgrid[0:height, 0:width]
    left_kidney = (
        ((xx - (x1 + (bbox_w * 0.34))) / max(bbox_w * 0.13, 1.0)) ** 2 +
        ((yy - (y1 + (bbox_h * 0.58))) / max(bbox_h * 0.17, 1.0)) ** 2
    ) <= 1.0
    right_kidney = (
        ((xx - (x1 + (bbox_w * 0.66))) / max(bbox_w * 0.13, 1.0)) ** 2 +
        ((yy - (y1 + (bbox_h * 0.58))) / max(bbox_h * 0.17, 1.0)) ** 2
    ) <= 1.0
    center_collecting = (
        ((xx - (x1 + (bbox_w * 0.50))) / max(bbox_w * 0.20, 1.0)) ** 2 +
        ((yy - (y1 + (bbox_h * 0.58))) / max(bbox_h * 0.22, 1.0)) ** 2
    ) <= 1.0

    return (left_kidney | right_kidney | center_collecting).astype(np.float32)


def build_gradcam_overlay(original, attention_map, reference_mode, disease_type=None):
    width, height = original.size
    normalized_map = resize_attention_map(attention_map, width, height)
    disease_type = str(disease_type or '').strip().lower()

    gray_arr = np.asarray(ImageOps.autocontrast(original.convert('L')), dtype=np.float32)
    anatomy_bbox = estimate_anatomy_bbox(gray_arr, disease_type=disease_type)
    anatomy_mask = build_soft_bbox_mask(width, height, anatomy_bbox, blur_ratio=0.050)
    normalized_map = normalized_map * (0.22 + (0.78 * anatomy_mask))

    # X-quang phoi: gioi han heatmap vao vung phe truoc khi phu mau.
    if disease_type == 'pneumonia_image':
        lung_mask = build_pneumonia_lung_mask(width, height, anatomy_bbox=anatomy_bbox)
        normalized_map = normalized_map * lung_mask
    elif disease_type == 'kidney_stone_image':
        kidney_mask = build_kidney_focus_mask(width, height, anatomy_bbox=anatomy_bbox)
        normalized_map = normalized_map * (0.35 + (0.65 * kidney_mask))

    normalized_map -= float(normalized_map.min())
    normalized_map /= max(float(normalized_map.max()), 1e-6)

    # Remove weak activations so overlay does not tint the entire image.
    focus_floor = 0.58 if reference_mode else 0.42
    focus = np.clip((normalized_map - focus_floor) / max(1.0 - focus_floor, 1e-6), 0.0, 1.0)
    focus = np.power(focus, 0.85 if reference_mode else 0.70)

    if np.count_nonzero(focus > 0) < max(20, int(width * height * 0.0015)):
        top_k = max(1, int(width * height * 0.035))
        flat = normalized_map.reshape(-1)
        threshold = np.partition(flat, -top_k)[-top_k]
        focus = np.clip((normalized_map - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)

    colored_map = colorize_attention_map(normalized_map, reference_mode)
    original_arr = np.asarray(original.convert('RGB'), dtype=np.float32) / 255.0

    intensity_cap = 0.38 if reference_mode else 0.70
    intensity = np.clip(focus[..., None] * intensity_cap, 0.0, intensity_cap)
    blended = np.clip((original_arr * (1.0 - intensity)) + (colored_map * intensity), 0.0, 1.0)
    return Image.fromarray((blended * 255.0).astype(np.uint8))


VISUALIZATION_FONT = None


def get_visualization_font(font_size):
    global VISUALIZATION_FONT

    candidates = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        'C:/Windows/Fonts/segoeui.ttf'
    ]

    if VISUALIZATION_FONT is None:
        VISUALIZATION_FONT = False
        for font_path in candidates:
            if os.path.exists(font_path):
                VISUALIZATION_FONT = font_path
                break

    if VISUALIZATION_FONT:
        try:
            return ImageFont.truetype(VISUALIZATION_FONT, font_size)
        except Exception:
            pass

    return ImageFont.load_default()


def find_connected_regions(mask, min_area):
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    regions = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy

                y_start = max(0, cy - 1)
                y_end = min(height, cy + 2)
                x_start = max(0, cx - 1)
                x_end = min(width, cx + 2)

                for ny in range(y_start, y_end):
                    for nx in range(x_start, x_end):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if area >= min_area:
                regions.append({
                    'x': int(min_x),
                    'y': int(min_y),
                    'w': int(max_x - min_x + 1),
                    'h': int(max_y - min_y + 1),
                    'area': int(area)
                })

    return regions


def rect_intersection_area(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h


def rect_iou(rect_a, rect_b):
    inter = rect_intersection_area(rect_a, rect_b)
    if inter <= 0:
        return 0.0

    area_a = max(1, (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1]))
    area_b = max(1, (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def region_to_rect(region):
    return (
        int(region['x']),
        int(region['y']),
        int(region['x'] + region['w']),
        int(region['y'] + region['h'])
    )


def region_center(region):
    return (
        float(region['x'] + (region['w'] / 2.0)),
        float(region['y'] + (region['h'] / 2.0))
    )


def suppress_close_regions(regions, max_regions=2, iou_threshold=0.20):
    if not regions:
        return []

    picked = []

    for candidate in regions:
        cand_rect = region_to_rect(candidate)
        cand_center = region_center(candidate)
        cand_scale = max(1.0, float(min(candidate['w'], candidate['h'])))
        should_skip = False

        for chosen in picked:
            chosen_rect = region_to_rect(chosen)
            if rect_iou(cand_rect, chosen_rect) >= iou_threshold:
                should_skip = True
                break

            chosen_center = region_center(chosen)
            center_distance = np.sqrt(
                ((cand_center[0] - chosen_center[0]) ** 2) +
                ((cand_center[1] - chosen_center[1]) ** 2)
            )
            chosen_scale = max(1.0, float(min(chosen['w'], chosen['h'])))
            if center_distance < (min(cand_scale, chosen_scale) * 0.75):
                should_skip = True
                break

        if should_skip:
            continue

        picked.append(candidate)
        if len(picked) >= max_regions:
            break

    if not picked:
        return [regions[0]]

    return picked


def rects_overlap(rect_a, rect_b, margin=0):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return not (
        (ax2 + margin) <= bx1 or
        (bx2 + margin) <= ax1 or
        (ay2 + margin) <= by1 or
        (by2 + margin) <= ay1
    )


def clamp_rect_to_image(left, top, width, height, image_width, image_height):
    width = min(width, image_width)
    height = min(height, image_height)
    left = int(max(0, min(left, image_width - width)))
    top = int(max(0, min(top, image_height - height)))
    return left, top, left + width, top + height


def find_label_rect_for_region(region_rect, label_width, label_height, image_width, image_height, occupied_label_rects):
    x1, y1, x2, y2 = region_rect
    margin = 4

    candidate_origins = [
        (x1, y1 - label_height - margin),
        (x1, y2 + margin),
        (x2 - label_width, y1 - label_height - margin),
        (x2 - label_width, y2 + margin)
    ]

    for left, top in candidate_origins:
        rect = clamp_rect_to_image(left, top, label_width, label_height, image_width, image_height)
        if any(rects_overlap(rect, occ, margin=2) for occ in occupied_label_rects):
            continue
        return rect

    step = max(8, int(label_height * 0.6))
    base_left = max(0, min(x1, image_width - label_width))
    for offset in range(0, image_height, step):
        for direction in (-1, 1):
            top = y1 + (offset * direction)
            rect = clamp_rect_to_image(base_left, top, label_width, label_height, image_width, image_height)
            if any(rects_overlap(rect, occ, margin=2) for occ in occupied_label_rects):
                continue
            return rect

    return clamp_rect_to_image(base_left, y1, label_width, label_height, image_width, image_height)


def build_fallback_region(arr, focus_mask, relative_size):
    height, width = arr.shape
    if not np.any(focus_mask):
        focus_mask = np.ones_like(arr, dtype=bool)

    focus_values = np.where(focus_mask, arr, -1)
    peak_y, peak_x = np.unravel_index(np.argmax(focus_values), focus_values.shape)

    box_size = max(24, int(min(height, width) * relative_size))
    half = box_size // 2
    x = max(0, int(peak_x) - half)
    y = max(0, int(peak_y) - half)
    w = min(width - x, box_size)
    h = min(height - y, box_size)

    return {
        'x': int(x),
        'y': int(y),
        'w': int(w),
        'h': int(h),
        'area': int(w * h),
        'score': 0.0
    }


def detect_kidney_stone_regions(gray_arr):
    height, width = gray_arr.shape
    smoothed = np.asarray(
        Image.fromarray(gray_arr.astype(np.uint8)).filter(ImageFilter.MedianFilter(size=5)),
        dtype=np.float32
    )

    anatomy_bbox = estimate_anatomy_bbox(smoothed, disease_type='kidney_stone_image')
    x1, y1, x2, y2 = anatomy_bbox
    yy, xx = np.mgrid[0:height, 0:width]
    margin_x = max(2, int((x2 - x1) * 0.06))
    margin_y = max(2, int((y2 - y1) * 0.08))
    center_mask = (
        (xx >= (x1 + margin_x)) & (xx <= (x2 - margin_x)) &
        (yy >= (y1 + margin_y)) & (yy <= (y2 - margin_y))
    )
    if np.count_nonzero(center_mask) < max(100, int(gray_arr.size * 0.02)):
        center_mask = (
            (xx >= width * 0.12) & (xx <= width * 0.88) &
            (yy >= height * 0.12) & (yy <= height * 0.88)
        )

    focus = smoothed[center_mask]
    if focus.size == 0:
        focus = smoothed.reshape(-1)
    threshold = max(np.percentile(focus, 99.4), float(focus.mean() + (1.8 * focus.std())))
    candidates = (smoothed >= threshold) & center_mask
    min_area = max(14, int(gray_arr.size * 0.00015))
    regions = find_connected_regions(candidates, min_area)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    for region in regions:
        x = region['x']
        y = region['y']
        w = region['w']
        h = region['h']
        patch = smoothed[y:y + h, x:x + w]
        intensity = float(patch.mean())
        region_cx = x + (w / 2.0)
        region_cy = y + (h / 2.0)
        center_distance = np.sqrt(
            ((region_cx - center_x) / max(center_x, 1.0)) ** 2 +
            ((region_cy - center_y) / max(center_y, 1.0)) ** 2
        )
        region['score'] = intensity + (region['area'] * 0.12) - (center_distance * 28.0)

    if not regions:
        regions = [build_fallback_region(smoothed, center_mask, 0.08)]

    regions.sort(key=lambda item: item.get('score', 0.0), reverse=True)
    return suppress_close_regions(regions, max_regions=2, iou_threshold=0.15)


def detect_pneumonia_regions(gray_arr):
    height, width = gray_arr.shape
    base_img = ImageOps.autocontrast(Image.fromarray(gray_arr.astype(np.uint8)))
    enhanced = np.asarray(base_img, dtype=np.float32)
    blurred = np.asarray(base_img.filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32)
    opacity = enhanced - blurred

    anatomy_bbox = estimate_anatomy_bbox(enhanced, disease_type='pneumonia_image')
    lung_mask = build_pneumonia_lung_mask(width, height, anatomy_bbox=anatomy_bbox) > 0.5
    soft_anatomy = build_soft_bbox_mask(width, height, anatomy_bbox, blur_ratio=0.045)
    lung_mask = lung_mask & (soft_anatomy > 0.18)

    lung_pixels = enhanced[lung_mask]
    opacity_pixels = opacity[lung_mask]
    if lung_pixels.size == 0 or opacity_pixels.size == 0:
        lung_mask = np.ones((height, width), dtype=bool)
        lung_pixels = enhanced[lung_mask]
        opacity_pixels = opacity[lung_mask]

    bright_threshold = max(np.percentile(lung_pixels, 72), lung_pixels.mean() + 0.35 * lung_pixels.std())
    opacity_threshold = opacity_pixels.mean() + 0.20 * opacity_pixels.std()

    candidates = (enhanced >= bright_threshold) & (opacity >= opacity_threshold) & lung_mask
    min_area = max(90, int(gray_arr.size * 0.0010))
    regions = find_connected_regions(candidates, min_area)

    for region in regions:
        x = region['x']
        y = region['y']
        w = region['w']
        h = region['h']
        patch = enhanced[y:y + h, x:x + w]
        opacity_patch = opacity[y:y + h, x:x + w]
        region['score'] = (region['area'] * 0.9) + float(patch.mean()) + (float(opacity_patch.mean()) * 2.0)

    if not regions:
        regions = [build_fallback_region(enhanced, lung_mask, 0.18)]

    regions.sort(key=lambda item: item.get('score', 0.0), reverse=True)
    return suppress_close_regions(regions, max_regions=2, iou_threshold=0.20)


def annotate_regions(image_bytes, disease_type, probability, decision_threshold, attention_map=None):
    original = resize_image_for_visualization(Image.open(BytesIO(image_bytes)).convert('RGB'))
    original_gray = np.asarray(ImageOps.autocontrast(original.convert('L')), dtype=np.float32)
    is_reference_mode = probability < decision_threshold

    if attention_map is not None:
        annotated = build_gradcam_overlay(original, attention_map, is_reference_mode, disease_type=disease_type)
        if disease_type == 'kidney_stone_image':
            if is_reference_mode:
                note = 'Lớp phủ xanh là heatmap Grad-CAM cho thấy vùng CNN chú ý trên ảnh CT. Mức nguy cơ hiện tại đang thấp, nên phần hiển thị này chỉ để tham khảo.'
            else:
                note = 'Lớp phủ nóng là heatmap Grad-CAM cho thấy vùng CNN chú ý nhiều nhất trên ảnh CT khi đánh giá nguy cơ sỏi thận. Đây là gợi ý hình ảnh, không thay thế kết luận của bác sĩ.'
        else:
            if is_reference_mode:
                note = 'Lớp phủ xanh là heatmap Grad-CAM cho thấy vùng CNN chú ý trên ảnh X-quang. Mức nguy cơ hiện tại đang thấp, nên phần hiển thị này chỉ để tham khảo.'
            else:
                note = 'Lớp phủ nóng là heatmap Grad-CAM cho thấy vùng CNN chú ý nhiều nhất trên ảnh X-quang khi đánh giá nguy cơ viêm phổi. Đây là gợi ý hình ảnh, không thay thế kết luận của bác sĩ.'

        visualization_mode = 'reference' if is_reference_mode else 'warning'
        return image_to_base64(original), image_to_base64(annotated), note, visualization_mode

    if disease_type == 'kidney_stone_image':
        regions = detect_kidney_stone_regions(original_gray)
        if is_reference_mode:
            title = 'Vùng tham khảo sỏi thận'
            note = 'Khung xanh là vùng hệ thống đánh dấu để tham khảo trên ảnh CT. Mức nguy cơ hiện tại đang thấp, nên các vùng này chỉ hỗ trợ quan sát và chưa đủ cơ sở để kết luận nguy cơ cao.'
            outline = (14, 116, 144, 255)
            fill = (14, 116, 144, 55)
        else:
            title = 'Vùng nghi ngờ sỏi thận'
            note = 'Khung đỏ là vùng sáng bất thường mà hệ thống AI đánh dấu trên ảnh CT. Đây là gợi ý hình ảnh, không thay thế kết luận của bác sĩ.'
            outline = (239, 68, 68, 255)
            fill = (239, 68, 68, 70)
    else:
        regions = detect_pneumonia_regions(original_gray)
        if is_reference_mode:
            title = 'Vùng tham khảo viêm phổi'
            note = 'Khung xanh là vùng hệ thống đánh dấu để tham khảo trên ảnh X-quang. Mức nguy cơ hiện tại đang thấp, nên các vùng này chỉ hỗ trợ quan sát và chưa đủ cơ sở để kết luận nguy cơ cao.'
            outline = (14, 116, 144, 255)
            fill = (14, 116, 144, 50)
        else:
            title = 'Vùng nghi ngờ viêm phổi'
            note = 'Khung vàng là vùng mờ bất thường trong trường phổi mà hệ thống AI đánh dấu trên ảnh X-quang. Đây là gợi ý hình ảnh, không thay thế kết luận của bác sĩ.'
            outline = (245, 158, 11, 255)
            fill = (245, 158, 11, 60)

    annotated = original.convert('RGBA')
    overlay = Image.new('RGBA', annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, int(min(original.size) * 0.008))
    font_size = max(14, int(min(original.size) * 0.028))
    font = get_visualization_font(font_size)
    label_padding_x = max(8, int(font_size * 0.45))
    label_padding_y = max(5, int(font_size * 0.30))
    image_width, image_height = original.size
    occupied_label_rects = []

    for index, region in enumerate(regions, start=1):
        x1 = region['x']
        y1 = region['y']
        x2 = region['x'] + region['w']
        y2 = region['y'] + region['h']
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=line_width, fill=fill)

        label_text = f'{title} #{index}'
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_height = text_height + (label_padding_y * 2)
        label_width = text_width + (label_padding_x * 2)
        label_left, label_top, label_right, label_bottom = find_label_rect_for_region(
            (x1, y1, x2, y2),
            int(label_width),
            int(label_height),
            image_width,
            image_height,
            occupied_label_rects
        )
        occupied_label_rects.append((label_left, label_top, label_right, label_bottom))
        draw.rectangle([label_left, label_top, label_right, label_bottom], fill=outline)
        draw.text((label_left + label_padding_x, label_top + label_padding_y - 1), label_text, fill=(255, 255, 255, 255), font=font)

    annotated = Image.alpha_composite(annotated, overlay).convert('RGB')

    visualization_mode = 'reference' if is_reference_mode else 'warning'

    return image_to_base64(original), image_to_base64(annotated), note, visualization_mode

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Health Disease Prediction API (Multi-disease)',
        'status': 'running',
        'models_loaded': list(MODELS.keys()),
        'image_models_loaded': list(IMAGE_MODELS.keys()),
        'version': '2.0',
        'endpoints': {
            '/predict': 'POST - Predict disease risk',
            '/predict-image': 'POST - Predict disease risk from medical image',
            '/diseases': 'GET - List available diseases',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS),
        'image_models_loaded': len(IMAGE_MODELS),
        'available_diseases': list(MODELS.keys()),
        'available_image_diseases': list(IMAGE_MODELS.keys())
    })

@app.route('/diseases', methods=['GET'])
def diseases():
    """List available diseases and their info"""
    return jsonify(DISEASE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease risk
    
    Expected JSON body:
    {
        "DiseaseType": "heart_disease|diabetes|hypertension|stroke",
        "Data": {
            // specific fields based on disease type
        }
    }
    
    Returns:
    {
        "Result": "Nguy cơ cao/trung bình/thấp",
        "RiskLevel": 75.5,
        "Recommendation": "...",
        "Details": "..."
    }
    """
    try:
        # Parse request
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must be JSON'
            }), 400
        
        # Get disease type (support both DiseaseType and disease_type)
        disease_type = request_data.get('DiseaseType') or request_data.get('disease_type', 'heart_disease')
        disease_type = disease_type.lower()
        disease_data = request_data.get('Data') or request_data.get('data') or request_data
        
        if disease_type not in MODELS:
            return jsonify({
                'error': f'Disease type "{disease_type}" not supported',
                'supported_types': list(MODELS.keys())
            }), 400
        
        print(f"\n📥 Prediction request for {disease_type}")
        print(f"   Data: {disease_data}")
        
        # Get model components
        model = MODELS[disease_type]
        scaler = SCALERS[disease_type]
        feature_names = FEATURE_SETS[disease_type]
        
        # Create input dataframe with robust feature mapping
        input_df = build_input_dataframe(disease_type, disease_data, feature_names, scaler)
        
        print(f"   Features: {list(input_df.columns)}")
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = None
        decision_threshold = get_decision_threshold(disease_type)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            probability = probabilities[1]  # Probability of class 1 (disease)
            prediction = 1 if probability >= decision_threshold else 0
        else:
            probability = 0.8 if prediction == 1 else 0.2
        
        # Get risk level and recommendation
        result, risk_level, recommendation, details = get_risk_level(probability, disease_type)
        
        # Create response
        response = {
            'DiseaseType': disease_type,
            'Result': result,
            'RiskLevel': float(risk_level),
            'Recommendation': recommendation,
            'Details': details,
            'Probability': float(probability),
            'PredictedClass': int(prediction),
            'DecisionThreshold': float(decision_threshold),
            'ModelType': str(type(model).__name__)
        }
        
        print(f"   ✅ Response: {result} ({risk_level:.1f}%)")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500


@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Predict disease risk from image file.

    Multipart form-data:
      - disease_type: kidney_stone_image|pneumonia_image
      - file: image binary
    """
    try:
        disease_type = (request.form.get('DiseaseType') or request.form.get('disease_type') or '').strip().lower()
        if not disease_type:
            return jsonify({'error': 'Missing disease_type'}), 400

        if disease_type not in IMAGE_MODELS:
            return jsonify({
                'error': f'Image disease type "{disease_type}" not supported',
                'supported_types': list(IMAGE_MODELS.keys())
            }), 400

        image_file = request.files.get('file') or request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'Missing image file. Use field name "file".'}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({'error': 'Image file is empty'}), 400

        model = IMAGE_MODELS[disease_type]
        labels = IMAGE_LABELS.get(disease_type, [])
        decision_threshold = get_image_decision_threshold(disease_type)
        attention_map = None
        model_type = 'sklearn_legacy'

        if isinstance(model, dict) and model.get('type') == 'pytorch_cnn':
            model_type = str(model.get('type') or 'pytorch_cnn')
            image_size = IMAGE_META.get(disease_type, {}).get('image_size', [224, 224])
            probability, attention_map = predict_probability_and_gradcam(
                model['model'],
                image_bytes,
                image_size=image_size,
                target_layer_name=model.get('target_layer', 'layer4'),
                device=model.get('device'),
            )
            pred_class = 1 if probability >= decision_threshold else 0
        else:
            model_type = str(type(model).__name__)
            feats = extract_image_features(image_bytes, disease_type)
            pred_class = int(model.predict(feats)[0])

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feats)[0]
                pos_idx = get_positive_class_index(disease_type, labels)
                probability = float(proba[pos_idx])
                pred_class = 1 if probability >= decision_threshold else 0
            else:
                probability = 0.8 if pred_class == 1 else 0.2

        result, risk_level, recommendation, details = get_risk_level(probability, disease_type)
        original_image_base64, annotated_image_base64, visualization_note, visualization_mode = annotate_regions(
            image_bytes,
            disease_type,
            probability,
            decision_threshold,
            attention_map=attention_map
        )

        response = {
            'DiseaseType': disease_type,
            'Result': result,
            'RiskLevel': float(risk_level),
            'Recommendation': recommendation,
            'Details': details,
            'OriginalImageBase64': original_image_base64,
            'AnnotatedImageBase64': annotated_image_base64,
            'VisualizationNote': visualization_note,
            'VisualizationMode': visualization_mode,
            'Probability': float(probability),
            'PredictedClass': int(pred_class),
            'DecisionThreshold': float(decision_threshold),
            'ModelType': model_type
        }

        return jsonify(response), 200
    except Exception as e:
        print(f"❌ Image prediction error: {str(e)}")
        return jsonify({'error': str(e), 'message': 'Image prediction failed'}), 500

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING MULTI-DISEASE PREDICTION API SERVER")
    print("="*70)
    print(f"\n✅ Available Models: {len(MODELS)}")
    for disease in MODELS.keys():
        disease_name = DISEASE_INFO.get(disease, {}).get('name', disease)
        print(f"   - {disease_name}")
    
    print(f"\n🔗 Server: http://localhost:5000")
    print("\n📍 Endpoints:")
    print("   GET  /              - API info")
    print("   GET  /health        - Health check")
    print("   GET  /diseases      - List available diseases")
    print("   POST /predict       - Disease prediction")
    print("   POST /predict-image - Disease prediction from image")
    
    print("\n💡 Example POST /predict:")
    print("""
    {
        "DiseaseType": "heart_disease",
        "Data": {
            "age": 55,
            "sex": "Male",
            "chestpaintype": 2,
            "restingbp": 140,
            "cholesterol": 250,
            "fastingbs": 1,
            "maxhr": 150,
            "exerciseangina": 0
        }
    }
    """)
    
    print("\n🔐 Configuration (appsettings.json):")
    print("""
    "AISettings": {
        "IsEnabled": true,
        "ApiUrl": "http://localhost:5000/predict"
    }
    """)
    
    print("\n" + "="*70 + "\n")
    
    # Run server
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=debug_mode, use_reloader=debug_mode)

