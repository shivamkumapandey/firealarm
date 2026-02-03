import cv2
import numpy as np
import threading
import time
import os
from flask import Flask, render_template, Response

# Initialize Flask
app = Flask(__name__)

# --- Configuration ---
ALARM_SOUND_FILE = "fire_alarm.mp3"  # Name of your sound file
USE_ALARM = True

# --- Sound Handling (Cross-Platform) ---
def play_alarm_sound():
    """Plays a sound. Uses playsound if available, otherwise system beep."""
    try:
        from playsound3 import playsound
        if os.path.exists(ALARM_SOUND_FILE):
            playsound(ALARM_SOUND_FILE)
        else:
            raise FileNotFoundError
    except Exception:
        # Fallback for Windows if file missing or lib not installed
        import winsound
        winsound.Beep(2500, 1000)

# --- Fire Alarm System Class ---
class FireAlarmSystem:
    def __init__(self):
        self.alarm_active = False
        self.last_detected_time = 0
        self.stop_thread = False
        # Threading ensures the video doesn't freeze when sound plays
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        while not self.stop_thread:
            if self.alarm_active:
                play_alarm_sound()
            else:
                time.sleep(0.1) # Check every 100ms

    def trigger(self):
        self.alarm_active = True
        self.last_detected_time = time.time()

    def update_status(self):
        # Debounce: Turn off alarm if fire not seen for 3 seconds
        if time.time() - self.last_detected_time > 3:
            self.alarm_active = False

alarm_system = FireAlarmSystem()

# --- Camera Handling ---
# '0' is usually the default webcam. Change to '1' if you have an external cam.
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # 1. Image Pre-processing (Blurring reduces noise)
        frame = cv2.resize(frame, (640, 480))
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # 2. Define colors for Fire (Orange/Yellow range)
        lower = np.array([18, 50, 50], dtype="uint8")
        upper = np.array([35, 255, 255], dtype="uint8")

        # 3. Create Mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # 4. Find Contours (Shapes)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected_this_frame = False

        for contour in contours:
            # Filter out small irrelevant orange spots (noise)
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Add text label
                cv2.putText(frame, "FIRE DETECTED", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                fire_detected_this_frame = True

        # 5. Update Alarm Logic
        if fire_detected_this_frame:
            alarm_system.trigger()
            # Visual Warning on Screen
            cv2.putText(frame, "!!! WARNING: ALARM ACTIVE !!!", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        alarm_system.update_status()

        # 6. Encode frame for Web Browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        # Cleanup when app stops
        camera.release()