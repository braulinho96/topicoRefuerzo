from utils.logger import Logger
from Simulations.PayloadSimulation import PayloadSimulation
import unittest
import json


class TestLogger(unittest.TestCase):
    """
        Example: Store data output to str in case
                Reason: Object of type Thread is not JSON serializable

        Sample data output to log in str format
            {
                '_thread': <Thread(Thread-1, stopped 6154940416)>,
                '_is_running': False,
                '_request_queue': <queue.Queue object at 0x103437790>,
                '_futures': {},
                '_futures_lock': <unlocked _thread.lock object at 0x1034378a0>,
                'count': 1
            }
    """

    def test_logging_object_payload_simulated_unserilizable(self):
        req = PayloadSimulation()
        req.start()
        req.stop()
        _data = json.dumps(str(req.__dict__))  # for safe convert to string and store
        Logger.write_data(
            _class_name=PayloadSimulation.__class__.__name__,
            _id=id(req),
            _data=_data,
            _log_type='success'
        )

    """
        Example: Object data that can be serialized 
        Output:
           {'name': 'dummy', 'value':'99', 'time': 2323232, 'log_type": 'info'}
    """

    def test_logging_dictionary_dummy_serilizable(self):
        class Dummy:
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def get_value(self):
                return {'name': self.name, 'value': self.value}

        _object = Dummy(name='dummy', value='99')
        _data = _object.get_value()

        Logger.write_data(
            _class_name=Dummy.__class__.__name__,
            _id=id(_object),
            _data=_data,
            _log_type='info'
        )


if __name__ == "__main__":
    unittest.main()
