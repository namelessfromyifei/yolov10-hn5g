from flask import Flask
from ultralytics import YOLOv10
import threading
from global_vars import flag

app = Flask(__name__)

def process_data():
    print(f"started-------------------------------------------")
    model = YOLOv10('weights/yolov10b.pt')
    # model -> predictor -> build -> loaders
    model.predict(source='rtmp://192.168.4.56/live/test',
                  vid_stride=2,
                  stream_buffer=False,
                  output_stream=True,
                  output_stream_source="rtmp://192.168.4.56/live2/test",
                  conf=0.6,
                  show=True, verbose=False, save=False, save_txt=False, save_crop=False)
    print(f"stopped-------------------------------------------")

thread = None

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/start')
def start_stream():
    global thread, flag
    if thread is not None and thread.is_alive():
        print(f"thread alive？:::{thread.is_alive()}")
        return 'stream already started'
    if thread is None:
        flag[0] = True
        thread = threading.Thread(target=process_data)
        thread.start()
    else:
        print(f"thread alive？:::{thread.is_alive()}")
        thread = None
        return 'something error, please start again!'
    return 'start stream!'

@app.route('/stop')
def stop_stream():
    global thread, flag
    if thread is None:
        return 'please start stream first!'

    flag[0] = False
    thread = None
    return 'stop stream!'