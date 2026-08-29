import time
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from Simulations.RotationSimulation import RotationSimulation

if __name__ == "__main__":
    # Create an instance of RotationSimulation
    rotation_simulation = RotationSimulation()

    try:
        # Start the simulation
        rotation_simulation.start()

        future = rotation_simulation.send_request("all")
        result = future.result()
        print(f"Initial data: {result}")

        rotation_simulation.set_torque(np.array([0, 0, 0.002]))

        future = rotation_simulation.send_request("all")
        result = future.result()
        print(f"With torque: {result}")

        time.sleep(5)

        future = rotation_simulation.send_request("all")
        result = future.result()
        print(f"Final data: {result}")

        rotation_simulation.set_torque(np.array([0.0, 0.0, -0.002]))

        future = rotation_simulation.send_request("all")
        result = future.result()
        print(f"With torque: {result}")

        time.sleep(12.0711)

        future = rotation_simulation.send_request("all")
        result = future.result()
        print(f"Final data: {result}")

        rotation_simulation.set_torque(np.array([0.0, 0.0, 0.0]))

        future = rotation_simulation.send_request("all")
        result = future.result()
        print(f"With torque: {result}")

    finally:
        # Stop the simulation
        rotation_simulation.stop()
