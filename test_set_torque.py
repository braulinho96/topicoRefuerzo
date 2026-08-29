from dotenv import load_dotenv
load_dotenv()

import numpy as np
import time
from Simulations.RotationSimulation import RotationSimulation
from SatellitePersonality import SatellitePersonality

def test_set_torque_application():
    """
    Tests the application of torque to the satellite's rotation simulation.

    This test verifies that applying a torque using `set_torque` results in the
    expected change in angular velocity, according to Euler's equations of motion.
    """
    print("=" * 60)
    print("Starting test: Torque Application")
    print("=" * 60)

    # 1. Initialize RotationSimulation
    rotation_sim = RotationSimulation(debug=False)
    print("RotationSimulation initialized.")

    # Set initial conditions: zero angular velocity for a simple case
    initial_angular_velocity = np.array([0.0, 0.0, 0.0])
    rotation_sim.angular_velocity = initial_angular_velocity
    print(f"Initial angular velocity set to: {initial_angular_velocity}")

    # 2. Start the simulation
    try:
        rotation_sim.start()
        print("Rotation simulation started.")

        # 3. Define test parameters
        test_torque = np.array([0.001, 0.0, 0.0])  # Apply torque on X-axis
        duration = 1.0  # seconds
        print(f"Applying torque {test_torque} for {duration} second(s).")

        # 4. Apply torque
        rotation_sim.set_torque(test_torque)
        time.sleep(duration)

        # Stop applying torque to measure the final state
        rotation_sim.set_torque(np.array([0.0, 0.0, 0.0]))
        time.sleep(0.1) # Give it a moment to stabilize

        # 5. Get the final angular velocity
        final_angular_velocity = rotation_sim.angular_velocity
        print(f"Final angular velocity measured: {final_angular_velocity}")

        # 6. Calculate expected change and verify
        # Simplified Euler's equation: d(omega)/dt = I_inv * tau
        # For a short duration and small omega, omega_final ~= I_inv * tau * dt
        inertia_matrix = np.diag(SatellitePersonality.MOMENT_OF_INERTIA)
        inertia_inv = np.linalg.inv(inertia_matrix)
        expected_delta_omega = inertia_inv @ test_torque * duration
        expected_final_omega = initial_angular_velocity + expected_delta_omega

        print(f"Expected final angular velocity: {expected_final_omega}")

        # Verification
        # Use a tolerance to account for simulation time steps and numerical precision
        tolerance = 1e-3
        is_close = np.allclose(final_angular_velocity, expected_final_omega, atol=tolerance)

        if is_close:
            print("\n[SUCCESS] The final angular velocity is close to the expected value.")
        else:
            print("\n[FAILURE] The final angular velocity differs significantly from the expected value.")
        
        error = np.linalg.norm(final_angular_velocity - expected_final_omega)
        print(f"Difference (norm of error): {error:.6f}")


    except Exception as e:
        print(f"An error occurred during the test: {e}")
    finally:
        # 7. Clean up
        if rotation_sim and rotation_sim._is_running:
            rotation_sim.stop()
            print("Rotation simulation stopped.")
        print("\nTest finished.")

if __name__ == "__main__":
    test_set_torque_application()
