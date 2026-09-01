import openvino as ov
from pathlib import Path

BASE_DIR = Path(__file__).parent
OV_MODELS_DIR = BASE_DIR / "models" / "openvino"

MODELS = {
    "yolo": str(OV_MODELS_DIR / "yolov5n_face.xml"),
    "arcface": str(OV_MODELS_DIR / "arcface.xml"),
    "occlusion": str(OV_MODELS_DIR / "occlusion_mobilenetv3.xml")
}
TARGET_EMB_PATH = str(BASE_DIR / "target_emb.npy")

# Model thresholds
THRESHOLD_TARGET = 0.30
THRESHOLD_UNSAFE_PERCENT = 30.0


def get_core():
    core = ov.Core()
    return core