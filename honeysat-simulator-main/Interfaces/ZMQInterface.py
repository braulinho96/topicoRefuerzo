from Interfaces.Interface import Interface
from SatStates.SatState import SatState
from SatStates.SuchaiState import SuchaiState
import zmq


class ZMQInterface(Interface):

    def __init__(self):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind("tcp://*:5555")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def request(self, timeout_ms: int = 100) -> bytes:
        """
        Read a request from the flight software
        :param timeout_ms: timeout in milliseconds
        :return: the request as a bytearray, or None if timeout expired
        """
        socks = dict(self.poller.poll(timeout_ms))
        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            request = self.socket.recv()
            print(f"Received request: {request}")
        else:
            request = None
        return request

    def reply(self, reply: bytes) -> None:
        """
        Send a reply to the flight software
        :param reply: Reply as a bytearray
        """
        print(f"Sending reply: {reply}")
        self.socket.send(reply)

    def server(self, satellite) -> None:
        try:
            while True:
                print("Server listening")
                request = self.socket.recv()
                print(f"Received request: {request}")
                reply = satellite.process_request(request)
                print(f"Sending reply: {reply}")
                self.socket.send(reply)
        except Exception as e:
            raise e
        finally:
            self.socket.close()
            self.context.destroy()
