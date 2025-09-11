import concurrent.futures
from abc import ABC, abstractmethod
import threading
import queue
from Simulations.SimulationRequest import SimulationRequest
from uuid import UUID
from threading import Lock


class Simulation(ABC):
    def __init__(self):
        self._thread = threading.Thread(target=self._run_wrapper)
        self._is_running = False
        self._request_queue = queue.Queue()
        self._futures = {}
        self._futures_lock = Lock()

    def _run_wrapper(self) -> None:
        while self._is_running:
            self._run_simulation()

    @abstractmethod
    def _run_simulation(self):
        """
        Each subclass should implement this method with its specific simulation logic.
        This method will be executed in a separate thread.
        """
        pass

    def start(self) -> None:
        """
        Starts the simulation by setting the running flag and starting the thread.
        """
        self._is_running = True
        self._thread.start()

    def stop(self) -> None:
        """
        Stops the simulation by clearing the running flag and joining the thread.
        """
        self._is_running = False
        self._thread.join()

    def send_request(self, request: str) -> concurrent.futures.Future:
        new_simulation_request = SimulationRequest(request)
        future = concurrent.futures.Future()

        with self._futures_lock:
            if new_simulation_request.id not in self._futures:
                self._futures[new_simulation_request.id] = future
            else:
                raise Exception(f"Request ID {new_simulation_request.id} already exists.")

        self._request_queue.put(new_simulation_request)

        return future

    def _set_result(self, request_id: UUID, result) -> None:
        with self._futures_lock:
            future = self._futures.get(request_id)
            if future:
                future.set_result(result)
                self._futures.pop(request_id)

    def _set_exception(self, request_id: UUID, exception) -> None:
        with self._futures_lock:
            future = self._futures.get(request_id)
            if future:
                future.set_exception(exception)
                self._futures.pop(request_id)
