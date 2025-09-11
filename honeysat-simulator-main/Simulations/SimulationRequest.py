import uuid


class SimulationRequest:
    def __init__(self, data):
        self.id = uuid.uuid4()
        self.data = data
