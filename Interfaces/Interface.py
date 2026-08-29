from abc import ABC, abstractmethod


class Interface(ABC):
    def __init__(self):
        """
        Initialize the communication Interface. This method should be overridden by all subclasses. The communication
        Interface, interfaces the HoneySat API with the Support Library Implemented on the Flight Software
        """

        pass

    @abstractmethod
    def request(self, timeout_ms: int) -> bytes:
        """
        Read a request from the flight software
        :param timeout_ms: timeout in milliseconds
        :return: the request as a bytearray, or None if timeout expired
        """
        pass

    @abstractmethod
    def reply(self, reply: bytes) -> None:
        """
        Send a reply to the flight software
        :param reply: Reply as a bytearray
        """
        pass
