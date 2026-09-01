import numpy as np
import cv2
from detector import OpenVINOModel 
import config

class SafetyAnalyzer:
    def __init__(self, occ_path, core):
        self.occ_model = OpenVINOModel(occ_path, core)

    def check_safety(self, face_224):
        f_128 = cv2.resize(face_224, (128, 128))
        o_input = (f_128.astype(np.float32) / 255.0).transpose((2, 0, 1))[np.newaxis, ...]
        occ_logits = self.occ_model.predict(o_input)[0]

        occ_probs = np.exp(occ_logits) / np.sum(np.exp(occ_logits))
        perc_coperto = (occ_probs[1] + occ_probs[2]) * 100

        is_unsafe = perc_coperto > config.THRESHOLD_UNSAFE_PERCENT
        return is_unsafe, perc_coperto
