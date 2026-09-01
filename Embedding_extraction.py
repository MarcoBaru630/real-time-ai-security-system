import glob, os, cv2, numpy as np
import openvino as ov
from detector import YoloDetector
from arcface_recognition import ArcFaceRecognizer

def run_extraction():
    core = ov.Core()
    detector = YoloDetector("models/openvino/yolov5n_face.xml", core=core)
    recognizer = ArcFaceRecognizer("models/openvino/arcface.xml", core=core)
    
    paths = glob.glob("target_images/*.*")
    embs = []

    for p in paths:
        img = cv2.imread(p)
        face = detector.get_biggest_face(img)
        if face is not None:
            embs.append(recognizer.embedding(face))
            print(f"Done: {p}")

    if embs:
        template = np.mean(np.stack(embs), axis=0)
        template /= np.linalg.norm(template)
        np.save("target_emb.npy", template)
        print("Embedding extracted")

if __name__ == "__main__":
    run_extraction()