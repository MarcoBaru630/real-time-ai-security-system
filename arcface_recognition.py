import cv2
import numpy as np
import openvino as ov

class ArcFaceRecognizer:
    def __init__(self, model_path, core=None, rgb_input=True):
        self.core = core if core else ov.Core()
        model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model, "CPU")
        self.output_layer = self.compiled_model.output(0)
        self.rgb_input = rgb_input
        self.target_emb = None

    def _l2_normalize(self, x, eps=1e-12):
        return x / max(np.linalg.norm(x), eps)

    def embedding(self, face_bgr):
        face = cv2.resize(face_bgr, (112, 112))
        if self.rgb_input:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))[np.newaxis, ...]
        results = self.compiled_model([face])[self.output_layer]
        return self._l2_normalize(results[0].astype(np.float32))

    def set_target(self, target_emb):
        self.target_emb = self._l2_normalize(target_emb.astype(np.float32))

    def is_target(self, face_bgr, threshold=0.3):
        if self.target_emb is None: return False, 0.0
        emb = self.embedding(face_bgr)
        sim = float(np.dot(emb, self.target_emb))
        return sim >= threshold, sim