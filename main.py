import cv2
import numpy as np
import threading
import time
import os
from camera import VideoStream
from detector import YoloDetector
from arcface_recognition import ArcFaceRecognizer
from Cover import SafetyAnalyzer
import config



def ai_worker(vs, detector, analyzer, recognizer, results_dict):
    print("AI Worker")
    
    while not vs.stopped:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # 1. Face detection
        detections, scores = detector.detect(frame)
        new_results = []

  
        if len(detections) == 0:
            with results_dict["lock"]:
                results_dict["results"] = []
            time.sleep(0.03) # Pausa per non saturare la CPU
            continue

        for det in detections:
            x, y, w, h = det[:4].astype(int)
            
            H, W = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] < 20:
                continue

            # Occlusion detection
            f_224 = cv2.resize(face_crop, (224, 224))
            is_unsafe, perc_coperto = analyzer.check_safety(f_224)
            
            # Face recognition
            is_target, sim = recognizer.is_target(cv2.resize(f_224, (112, 112)), 
                                                threshold=config.THRESHOLD_TARGET)

            # Output palette
            identity_label = "TARGET" if is_target else "OTHER"
            safety_label = "UNSAFE" if is_unsafe else "SAFE"
            label = f"{identity_label} {safety_label}"

           
            if is_target:
                color = (0, 255, 0) if not is_unsafe else (0, 255, 255)
            else:
                color = (255, 0, 0) if not is_unsafe else (0, 0, 255)

            new_results.append({
                "box": [x1, y1, x2, y2], 
                "label": f"{label} ({sim:.2f})", 
                "color": color
            })

        with results_dict["lock"]:
            results_dict["results"] = new_results
        
        # Piccola pausa per bilanciare il carico sulla RPi5
        time.sleep(0.02)


def main():
    
    core = config.get_core()
    
    
    print("Model uploading...")
    detector = YoloDetector(config.MODELS["yolo"], core)
    analyzer = SafetyAnalyzer(config.MODELS["occlusion"], core)
    recognizer = ArcFaceRecognizer(config.MODELS["arcface"], core)

  
    if os.path.exists(config.TARGET_EMB_PATH):
        recognizer.set_target(np.load(config.TARGET_EMB_PATH))
        print(f"Target uploaded from: {config.TARGET_EMB_PATH}")
    else:
        print("WARNING: target_emb.npy not found. No TARGET.")

    
    vs = VideoStream().start()
    print("Camera avviata (Warmup 2s)...")
    time.sleep(2.0)

  
    results_data = {"results": [], "lock": threading.Lock()}
    
    
    thread_ai = threading.Thread(
        target=ai_worker, 
        args=(vs, detector, analyzer, recognizer, results_data), 
        daemon=True
    )
    thread_ai.start()


    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue

           
            with results_data["lock"]:
                for r in results_data["results"]:
                    x1, y1, x2, y2 = r["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), r["color"], 2)
                    cv2.putText(frame, r["label"], (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, r["color"], 2)

            
            cv2.imshow("Sacurity Detection", frame)
            
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("\nInterruption")
    finally:
        print("Syestem closed")
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
