import time
from queue import Empty
from random import random

from Simulations.Simulation import Simulation


class PayloadSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.count = 0

    def _run_simulation(self):
        try:
            self.count += 1
            time.sleep(random() * 2)  # Simulate some processing time
            request = self._request_queue.get(timeout=0.1)
            if request is None:
                return

            # Simulate processing the request
            response = f"Processed request {request.id}, count is now {self.count}"

            # Set the result for the corresponding future
            self._set_result(request.id, response)

        except Empty:
            # No more requests in the queue
            pass

        except Exception as e:
            # If an exception occurs, set it on the future
            if request and request.id in self._futures:
                self._set_exception(request.id, e)
            else:
                print(f"Exception in simulation: {e}")
