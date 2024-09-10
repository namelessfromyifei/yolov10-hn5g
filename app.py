from flask import Flask, request
from ultralytics import YOLOv10
import threading
from global_vars import flag
import time

app = Flask(__name__)

def process_data(source, output_stream_source, ai_type, sn):
    print(f"{sn} started-------------------------------------------")
    if type == '0':
        model = YOLOv10('weights/yolov10b.pt')
    elif type == '1':
        model = YOLOv10('weights/yolov10b.pt')
    elif type == '2':
        model = YOLOv10('weights/yolov10b.pt')
    elif type == '3':
        model = YOLOv10('weights/yolov10b.pt')
    else:
        model = YOLOv10('weights/yolov10b.pt')

    # model -> predictor -> build -> loaders
    print(f"source: {source}, output_stream_source: {output_stream_source}, type: {ai_type}, sn: {sn}")
    model.predict(source=source,
                  vid_stride=25,
                  stream_buffer=False,
                  output_stream=True,
                  output_stream_source=output_stream_source,
                  ai_type=ai_type,
                  sn=sn,
                  conf=0.6,
                  show=True, verbose=False, save=False, save_txt=False, save_crop=False)
    # while flag[sn]:
    #     print(f"{sn} is running")
    #     time.sleep(1)
    print(f"{sn} stopped-------------------------------------------")

thread = {}

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/test', methods=['GET', 'POST'])
def test():
    print(request.args.get("source"))
    print(request.args.get("output_stream_source"))
    print(request.args.get("type"))
    return "test"

@app.route('/start', methods=['GET', 'POST'])
def start_stream():
    global thread, flag
    source = request.args.get("source")
    output_stream_source = request.args.get("output_stream_source")
    ai_type = request.args.get("type")
    sn = request.args.get("sn")
    if source is None or output_stream_source is None or ai_type is None or sn is None:
        return "need params {source,output_stream_source,type,sn}"
    if sn not in thread:
        thread[sn] = None
    if thread[sn] is not None and thread[sn].is_alive():
        print(f"thread {sn} alive？:::{thread[sn].is_alive()}")
        return f'{sn} stream already started'
    if thread[sn] is None:
        flag[sn] = True
        thread[sn] = threading.Thread(target=process_data,args=(source, output_stream_source, ai_type, sn))
        thread[sn].start()
    else:
        print(f"thread {sn} alive？:::{thread[sn].is_alive()}")
        thread[sn] = None
        return 'something error, please start again!'
    return 'start stream!'

@app.route('/stop', methods=['GET', 'POST'])
def stop_stream():
    global thread, flag
    sn = request.args.get("sn")
    if sn is None:
        return "need param {sn}"
    if sn not in thread or thread[sn] is None:
        return 'please start stream first!'

    flag[sn] = False
    thread[sn] = None
    return f'stop {sn} stream!'

if __name__ == '__main__':
    app.run(port=7777, debug=False)