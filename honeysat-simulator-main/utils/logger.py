import logging

from utils.db_client import MongoDBActor
from datetime import datetime

"""
    Input
    =====
        class_name:
            collection name to store data
            example: the name of the invoking class, or method
        id:
            unique identifier of the calling object Ex. id(my_object)
        data:
            The data associated to the class to store (object instance variable, or associated information)
        log_type: The basic logging type, following are the types.
            logging.INFO
            logging.DEBUG
            logging.ERROR
            logging.CRITICAL
            logging.WARNING or 
            custom_message
    
    How to consume this file from other classes?
    ============================================
    
    Step 1: Add the above import to the top of the class
    
    from utils.logger import Logger
    Logger(MyClass.__class__.__name__, json.dumps(my_class.__dict__) 
    
    
    Step 2: Where ever required to log any data, add the required sample log statement as shown below
    In this example below, the class PayLoadProcessor is invoked, and stores the serialized data of the class into Json
    
    >> Start 
        my_payload = PayLoadProcessor("argument_1", "argument_2", argument_3")
        _request_payload = my_payload.process_payload()
        if _request_payload == "success":
            log_type = "info" # or "success" as custom message 
        else:
            log_type = "error"
        
        Logger.write_data(
            _class_name = MyClass.__class__.__name__,
            _id = id(class_name),
            _data = json.dumps(my_payload.__dict__),
            _log_type = log_type
        )
    << End
"""


class Logger:
    @staticmethod
    def write_data(_class_name, _id, _data, _log_type):
        MongoDBActor(_class_name).insert_data({
            'id': _id,
            'data': _data,
            'log_type': _log_type,
            'time': datetime.now()
        })
