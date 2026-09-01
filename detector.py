import cv2
import numpy as np
import openvino as ov
class OpenVINOModel:
    def __init__(self, model_path, core=None, device="CPU"):
        self.core = core if core else ov.Core()
        model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, blob):
        return self.compiled_model([blob])[self.output_layer]
class YoloDetector:
    def __init__(self, model_path, core=None, conf_thres=0.4, iou_thres=0.45):
        self.core = core if core else ov.Core()
        model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = im.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2
        im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (left, top)

    def detect(self, bgr):
        img_lb, r, pad = self._letterbox(bgr, (640, 640))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        blob = img_rgb.transpose((2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        
        out = self.compiled_model([blob])[self.output_layer][0]
        scores = out[:, 4]
        keep = scores > self.conf_thres
        out, scores = out[keep], scores[keep]
        
        if out.shape[0] == 0: return np.empty((0, 4)), []

        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x1 = (cx - w / 2 - pad[0]) / r
        y1 = (cy - h / 2 - pad[1]) / r
        
        boxes = np.stack([x1, y1, w/r, h/r], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        
        if len(indices) > 0:
            return boxes[indices.flatten()], scores[indices.flatten()]
        return np.empty((0, 4)), []

    def get_biggest_face(self, img_bgr):
        boxes, _ = self.detect(img_bgr)
        if boxes.shape[0] == 0: return None
        areas = boxes[:, 2] * boxes[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = boxes[i].astype(int)
        H, W = img_bgr.shape[:2]
        return img_bgr[max(0, y):min(H, y+h), max(0, x):min(W, x+w)]
