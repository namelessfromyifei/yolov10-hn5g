from flask import Flask, request, jsonify
from ultralytics import YOLOv10
import threading
from global_vars import flag, announceDir
import time
from flask_sock import Sock
from MqttClient import MqttConnection
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
app.config['MQTT_BROKER_URL'] = 'broker.emqx.io'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_BROKER_USERNAME'] = ''
app.config['MQTT_BROKER_PASSWORD'] = ''
app.config['MQTT_KEEPALIVE'] = 60
app.config['MQTT_TLS_ENABLED'] = False
topic = 'hn5g/info'
mqtt_conn = MqttConnection()
mqtt_conn.connect(app)
mqtt_client = MqttConnection().client

def check_announce(sn):
    if sn not in announceDir:
        announceDir[sn] = {'target':'bottles','count':0, 'announce':False, 'msg':'bottles in video!!!!'}
    while flag[sn]:
        time.sleep(1)
        # print(announceDir[sn]['count'],"-",announceDir[sn]['target'])
        if announceDir[sn]['announce']:
            mqtt_client.publish(f'hn5g/info/{sn}', announceDir[sn]['msg'])

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

    thread_check = threading.Thread(target=check_announce, args=(sn, ))
    thread_check.start()

    # model -> predictor -> build -> loaders
    print(f"source: {source}, output_stream_source: {output_stream_source}, type: {ai_type}, sn: {sn}")
    model.predict(source=source,
                  vid_stride=20,
                  stream_buffer=False,
                  output_stream=True,
                  output_stream_source=output_stream_source,
                  ai_type=ai_type,
                  sn=sn,
                  conf=0.6,
                  show=False, verbose=False, save=False, save_txt=False, save_crop=False)
    # while flag[sn]:
    #     print(f"{sn} is running")
    #     time.sleep(1)
    print(f"{sn} stopped-------------------------------------------")

def return_stream_status(ws):
    keys_str = ""
    values_str = ""
    for key, value in flag.items():
        keys_str += f'{key: <10}'  # 调整占位宽度，10 可以根据实际情况修改
        value = "running" if value else "stopped"
        values_str += f'{value: <10}'

    result_str = keys_str + '\n' + values_str
    ws.send(f"{result_str}")

thread = {}
flag = flag
@app.route('/')
def hello_world():
    return 'Hello World!'

@sock.route('/status')
def status(ws):
    while True:
        time.sleep(1)
        return_stream_status(ws)
@app.route('/test', methods=['GET', 'POST'])
def test():
    print(request.args.get("source"))
    print(request.args.get("output_stream_source"))
    print(request.args.get("type"))
    return jsonify({"result": 1, "data": ""})

@app.route('/clear_announce')
def clear_announce():
    sn = request.args.get("sn")
    announceDir[sn]['announce'] = False
    return jsonify({"result": 1, "data": f"{sn} announce stopped!"})

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
    return jsonify({"result": 1, "data": "start stream!"})

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
    app.run(debug=True, host='0.0.0.0')