import cv2
import numpy as np
import threading
import time
import os
import sys
from flask import Flask, render_template, Response

# Initialize Flask
app = Flask(__name__)

# --- Configuration ---
ALARM_SOUND_FILE = "fire_alarm.mp3"

# --- Sound Handling ---
def play_alarm_sound():
    try:
        from playsound3 import playsound
        if os.path.exists(ALARM_SOUND_FILE):
            playsound(ALARM_SOUND_FILE)
    except Exception:
        import winsound
        winsound.Beep(2500, 1000)

# --- Fire Alarm Logic ---
class FireAlarmSystem:
    def __init__(self):
        self.alarm_active = False
        self.last_detected_time = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        while True:
            if self.alarm_active:
                play_alarm_sound()
            else:
                time.sleep(0.1)

    def trigger(self):
        self.alarm_active = True
        self.last_detected_time = time.time()

    def update_status(self):
        if time.time() - self.last_detected_time > 3:
            self.alarm_active = False

alarm_system = FireAlarmSystem()

# --- SMART CAMERA FINDER ---
def find_camera():
    """Tries to find a working camera index automatically."""
    print("📷 Searching for camera...")
    # Try indices 0, 1, and 2
    for index in range(3):
        # CAP_DSHOW is a Windows-specific flag that fixes many "black screen" issues
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            success, _ = cap.read()
            if success:
                print(f"✅ Camera found at Index {index}!")
                return cap
            else:
                cap.release()
    
    print("❌ ERROR: No working camera found. Please check connections.")
    return None

# Initialize Camera
camera = find_camera()

def generate_frames():
    global camera
    if camera is None:
        # If no camera, yield a blank error frame to prevent browser crash
        while True:
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "CAMERA NOT FOUND", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)

    while True:
        success, frame = camera.read()
        if not success:
            # Try to reconnect if camera disconnects
            camera.release()
            camera = find_camera()
            if camera is None:
                break
            continue

        # --- FIRE DETECTION LOGIC ---
        frame = cv2.resize(frame, (640, 480))
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower = np.array([18, 50, 50], dtype="uint8")
        upper = np.array([35, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE DETECTED", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                fire_detected = True

        if fire_detected:
            alarm_system.trigger()
            cv2.putText(frame, "!!! ALARM !!!", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        alarm_system.update_status()

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        # Host 0.0.0.0 allows access from other devices on the network
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if camera:
            camera.release()