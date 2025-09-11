from Simulations.PowerSystemSimulation import PowerSystemSimulation
from datetime import datetime, timezone, timedelta

if __name__ == "__main__":
    # Create an instance of OrbitalSimulation
    eps_simulation = PowerSystemSimulation()

    try:
        # Start the simulation
        eps_simulation.start()

        # Run the simulation for a certain duration (e.g., 60 seconds)
        simulation_duration = 10  # seconds
        end_time = datetime.utcnow() + timedelta(seconds=simulation_duration)

        while datetime.utcnow() < end_time:
            # Send a simulation request and get the future
            future = eps_simulation.send_request("Consume battery")

            # Simulate the passage of time
            eps_simulation._simulate_processing_time()

            # Wait for the simulation result
            result = future.result()
            print(result)

    finally:
        # Stop the simulation
        eps_simulation.stop()
