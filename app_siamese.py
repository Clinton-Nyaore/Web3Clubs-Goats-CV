from ultralytics import YOLO
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, Response, session, redirect, url_for
from flask_socketio import SocketIO
import yt_dlp as youtube_dl
from predict_siamese import similar_goat  # Import the similarity check function

# Load your object detection model
model_object_detection = YOLO("cabra_best.pt")

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
stop_flag = False

class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        print("*********************************Video Streaming******************************")
        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0
        self.known_goats = {
            "eth1": ["./goat_images/eth1/1.jpg", "./goat_images/eth1/2.jpg", "./goat_images/eth1/3.jpg", "./goat_images/eth1/4.jpg", "./goat_images/eth1/5.jpg"],
            "layer2": ["./goat_images/layer2/1.jpg", "./goat_images/layer2/2.jpg", "./goat_images/layer2/3.jpg", "./goat_images/layer2/4.jpg", "./goat_images/layer2/5.jpg"],
            "zk": ["./goat_images/zk/1.jpg", "./goat_images/zk/2.jpg", "./goat_images/zk/3.jpg", "./goat_images/zk/4.jpg", "./goat_images/zk/5.jpg"]
        }

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = int(value)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    def show(self, url):
        print(url)
        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best",
            "forceurl": True,
        }
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        info = ydl.extract_info(url, download=False)
        url = info["url"]

        cap = cv2.VideoCapture(url)
        while True:
            if self._preview:
                if stop_flag:
                    print("Process Stopped")
                    return

                grabbed, frame = cap.read()
                if not grabbed:
                    break
                if self.flipH:
                    frame = cv2.flip(frame, 1)
                if self.detect:
                    print("Confidence: ", self._confidence)
                    predictions = model_object_detection.predict(frame, conf=self._confidence / 100)

                    detected_labels = []
                    for prediction in predictions:
                        prediction.names = {0: 'goat'}
                        for box in prediction.boxes:
                            cropped_img = frame[int(box.top):int(box.bottom), int(box.left):int(box.right)]
                            for goat_name, goat_images in self.known_goats.items():
                                if similar_goat(cropped_img, goat_images):
                                    detected_labels.append(goat_name)
                                    break

                    if detected_labels:
                        frame = predictions[0].plot()
                        for label in detected_labels:
                            cv2.putText(frame, label, (int(box.left), int(box.top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            socketio.emit('label', label)
                    else:
                        frame = predictions[0].img

                frame = cv2.imencode(".jpg", frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                snap = np.zeros((1000, 1000), np.uint8)
                label = "Streaming Off"
                H, W = snap.shape
                font = cv2.FONT_HERSHEY_PLAIN
                color = (255, 255, 255)
                cv2.putText(snap, label, (W // 2 - 100, H // 2), font, 2, color, 2)
                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


VIDEO = VideoStreaming()

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('hompage.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    print("index")
    global stop_flag
    stop_flag = False
    if request.method == 'POST':
        print("Index post request")
        url = request.form['url']
        print("index: ", url)
        session['url'] = url
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    url = session.get('url', None)
    print("video feed: ", url)
    if url is None:
        return redirect(url_for('homepage'))
    return Response(VIDEO.show(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/request_preview_switch")
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    print("*" * 10, VIDEO.preview)
    return "nothing"

@app.route("/request_flipH_switch")
def request_flipH_switch():
    VIDEO.flipH = not VIDEO.flipH
    print("*" * 10, VIDEO.flipH)
    return "nothing"

@app.route("/request_run_model_switch")
def request_run_model_switch():
    VIDEO.detect = not VIDEO.detect
    print("*" * 10, VIDEO.detect)
    return "nothing"

@app.route('/update_slider_value', methods=['POST'])
def update_slider_value():
    slider_value = request.form['sliderValue']
    VIDEO.confidence = slider_value
    return 'OK'

@app.route('/stop_process')
def stop_process():
    print("Process stop Request")
    global stop_flag
    stop_flag = True
    return 'Process Stop Request'

@socketio.on('connect')
def test_connect():
    print('Connected')

if __name__ == "__main__":
    socketio.run(app, debug=True)
