
import cv2
import numpy as np
import subprocess
import threading

class VideoStream:
    def __init__(self):
        self.video_cmd = [
            "rpicam-vid", "-t", "0", "--inline", "--width", "640", "--height", "480",
            "--codec", "mjpeg", "-n", "--framerate", "30", "-o", "-"
        ]
        self.process = subprocess.Popen(self.video_cmd, stdout=subprocess.PIPE, bufsize=10**6)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        raw_bytes = b""
        while not self.stopped:
            chunk = self.process.stdout.read(8192)
            if not chunk: break
            raw_bytes += chunk
            a = raw_bytes.find(b'\xff\xd8')
            b = raw_bytes.find(b'\xff\xd9', a + 2) if a != -1 else -1
            if a != -1 and b != -1:
                jpg_data = raw_bytes[a:b+2]
                raw_bytes = raw_bytes[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with self.lock: self.frame = img

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.process.terminate()
