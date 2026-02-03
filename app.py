from flask import Flask, render_template, Response
import cv2
import numpy as np
import threading
import time
import winsound  # Fallback sound for Windows
import os

# Try importing playsound, handle if missing
try:
    from playsound3 import playsound
    SOUND_LIB_AVAILABLE = True
except ImportError:
    SOUND_LIB_AVAILABLE = False

app = Flask(__name__)

# --- Fire Alarm System Class ---
class FireAlarmSystem:
    def __init__(self):
        self.alarm_active = False
        self.stop_thread = False
        self.last_detected_time = 0
        # Background thread for audio to prevent video lag
        self.alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self.alarm_thread.start()

    def _alarm_loop(self):
        while not self.stop_thread:
            if self.alarm_active:
                if SOUND_LIB_AVAILABLE and os.path.exists('fire_alarm.mp3'):
                    try:
                        playsound('fire_alarm.mp3')
                    except:
                        self._fallback_beep()
                else:
                    self._fallback_beep()
            else:
                time.sleep(0.1)

    def _fallback_beep(self):
        winsound.Beep(2500, 1000)

    def trigger(self):
        self.alarm_active = True
        self.last_detected_time = time.time()

    def update(self):
        # Stop alarm if fire hasn't been seen for 3 seconds (debounce)
        if time.time() - self.last_detected_time > 3:
            self.alarm_active = False

# Initialize System
alarm_system = FireAlarmSystem()
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # 1. Resize & Blur
        frame = cv2.resize(frame, (640, 480))
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # 2. Fire Colors (Optimized)
        lower = np.array([18, 50, 50], dtype="uint8")
        upper = np.array([35, 255, 255], dtype="uint8")

        # 3. Masking
        mask = cv2.inRange(hsv, lower, upper)
        
        # 4. Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Threshold to avoid noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE DETECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                fire_detected = True

        # 5. Alarm Logic
        if fire_detected:
            alarm_system.trigger()
            cv2.putText(frame, "!!! WARNING: ALARM !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        alarm_system.update()

        # 6. Encode for Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream bytes to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)