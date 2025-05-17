from flask import Flask, render_template, Response
import cv2
import imutils

app = Flask(__name__)
camera = cv2.VideoCapture(0)
gun_cascade = cv2.CascadeClassifier('cascade.xml')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            guns = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
            for (x, y, w, h) in guns:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "GUN DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
