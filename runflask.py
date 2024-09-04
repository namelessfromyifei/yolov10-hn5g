from flask import Flask
from ultralytics import YOLOv10
import threading

app = Flask(__name__)

vid_stride = 2

def process_data():
    global vid_stride
    print(f"started--------vid_stride:{vid_stride}------------------------------------")
    model = YOLOv10('weights/yolov10b.pt')
    # model -> predictor -> build -> loaders
    model.predict(source='rtmp://127.0.0.1/live/test',
                  vid_stride=vid_stride,
                  stream_buffer=True,
                  conf=0.6,
                  show=True, verbose=False, save=False, save_txt=False, save_crop=False)
    print(f"stopped--------vid_stride:{vid_stride}------------------------------------")

thread = None

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/start')
def start_stream():
    global thread, vid_stride
    if thread is not None and thread.is_alive():
        print(f"thread alive？:::{thread.is_alive()}")
        return 'stream already started'
    if thread is None:
        vid_stride = 2
        thread = threading.Thread(target=process_data)
        thread.start()
    else:
        print(f"thread alive？:::{thread.is_alive()}")
        vid_stride = -1
        thread = None
        return 'something error, please start again!'
    return 'start stream!'

@app.route('/stop')
def stop_stream():
    global thread, vid_stride
    if thread is None:
        return 'please start stream first!'

    vid_stride = -1
    thread = None
    return 'stop stream!'