from flask_mqtt import Mqtt

class MqttConnection:
    _instance = None
    _client = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MqttConnection, cls).__new__(cls)
        return cls._instance

    def connect(self, app):
        if not self._client:
            self._client = Mqtt(app)

    @property
    def client(self):
        return self._client