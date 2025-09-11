class SensorData:
    def __init__(self, name: str, value: object, data_type=None):
        self.name = name
        self.type = data_type
        self.value = value
