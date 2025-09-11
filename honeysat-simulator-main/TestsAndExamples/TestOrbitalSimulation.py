from Simulations.OrbitalSimulation import OrbitalSimulation
from datetime import datetime, timezone, timedelta

from Simulations.RotationSimulation import RotationSimulation

if __name__ == "__main__":
    # Create an instance of OrbitalSimulation
    rotation_simulation = RotationSimulation()

    # Start the simulations
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    # orbit_simulation.print_method_return_types()
    # orbit_simulation.compare_updated_and_individual_methods()
    orbit_simulation.compare_orientation_methods()

    orbit_simulation.stop()
    rotation_simulation.stop()
