import os

class SatellitePersonality:
    # SUCHAI-2 PERSONALITY

    SATELLITE_NAME = os.environ['SATELLITE_NAME_TLE']

    # Orbital Simulation parameters
    NORAD_CATALOG_NUMBER = os.environ['SATELLITE_NORAD_CATALOG_NUMBER']
    # Use TLE_MAX_AGE to set how long before the Orbital Simulation downloads a new version of the TLE file
    TLE_MAX_AGE = 24  # in hours
    OBSERVER_LATITUDE = float(os.environ['GROUND_STATION_LAT'])  # Latitude of University of Chile
    OBSERVER_LONGITUDE = float(os.environ['GROUND_STATION_LON'])  # Longitude of University of Chile

    # ALTITUDE ANGLE THRESHOLD VALUES IN DEGREES (0 TO 90)
    THRESHOLD_1 = 5
    THRESHOLD_2 = 10
    THRESHOLD_3 = 45
    THRESHOLD_4 = 80

    # PACKET SUCCESS PROBABILITIES RELATIVE TO ALTITUDE ANGLE (0.0 TO 1.0)
    PROBABILITY_1 = 0.0  # LESS THAN 0 DEGREES OR BELOW HORIZON
    PROBABILITY_2 = 0.0  # BETWEEN 0 AND THRESHOLD 1
    PROBABILITY_3 = 0.95  # BETWEEN 1 AND THRESHOLD 2
    PROBABILITY_4 = 0.97  # BETWEEN 2 AND THRESHOLD 3
    PROBABILITY_5 = 0.98  # BETWEEN 3 AND THRESHOLD 4
    PROBABILITY_6 = 0.99  # BETWEEN 4 AND 90 DEGREES (MAX ALTITUDE ANGLE)

    # TEMPERATURE RANGE
    MIN_TEMPERATURE = 253  # Minimum temperature in Kelvin
    MAX_TEMPERATURE = 313  # Maximum temperature in Kelvin

    # Power System Simulation parameters
    NOMINAL_CAPACITY = 3.0  # Nominal Capacity of a single battery cell in Ah
    HEATER_POWER = 5.0  # Battery Heater power in Watts
    # TODO use this values in their corresponding sensors
    VOLTAGE_UNIT_MULTIPLIER = 1000
    CURRENT_UNIT_MULTIPLIER = 1000
    TEMPERATURE_UNIT_CHANGE = 273  # Subtract this value to convert to Celsius

    # Rotation Simulation parameters
    MOMENT_OF_INERTIA = (0.025, 0.025, 0.005)  # kg⋅m^2
    MAX_TORQUE_REACTION_WHEEL = 3.2 * pow(10, -3)  # N⋅m

    SOLAR_PANEL_ORIENTATION = (0, 0, 0)
    # SUN_SENSOR_ORIENTATION = (0, 0, np.pi / 2)
    SUN_SENSOR_ORIENTATION = (0, 0, 0)
    MULTIPLE_SUN_SENSOR_ORIENTATION = [
        (0, 0, 1),  # Front face, facing positive z-axis
        (0, 0, -1),  # Back face, facing negative z-axis
        (0, 1, 0),  # Top face, facing positive y-axis
        (0, -1, 0),  # Bottom face, facing negative y-axis
        (1, 0, 0),  # Right face, facing positive x-axis
        (-1, 0, 0)  # Left face, facing negative x-axis
    ]

    SUN_SENSOR_FIELD_OF_VIEW = 180  # in degrees
    SUN_SENSOR_MIN_CURRENT = 0  # in uA
    SUN_SENSOR_MAX_CURRENT = 930  # in uA

    # FLIGHT SOFTWARE DEVICE IDS AND ADDRESSES
    # On Board Computer (OBC)
    # OBC device ID
    SIM_OBC_ID = 0X01
    # Temperature
    SIM_OBC_ADDR_TEMP = 0X00

    # Electrical Power System (EPS)
    # EPS device ID
    SIM_EPS_ID = 0X02
    # Housekeeping (int vbat, int current_in, int current_out, int temp)
    SIM_EPS_ADDR_HKP = 0X00
    # Set ouput
    SIM_EPS_ADDR_SET = 0X01
    # Set header
    SIM_EPS_ADDR_HEATER = 0X02
    # Hard reset
    SIM_EPS_ADDR_RESET = 0X03

    # Attitude Determination And Control System (ADCS)
    # ADCS device ID
    SIM_ADCS_ID = 0X03
    # Magnetometer
    SIM_ADCS_ADDR_MAG = 0X00
    # Gyroscope
    SIM_ADCS_ADDR_GYR = 0X01
    # Sun sensor
    SIM_ADCS_ADDR_SUN = 0X02
    # Magnetorquer
    SIM_ADCS_ADDR_MTT = 0X03

    # Payload (RGB Camera)
    # Payload device ID
    SIM_CAMERA_ID = 0X04
    # Value to set some camera config
    SIM_CAM_ADDR_SIZE = 0X00
    # Picture Path
    SIM_CAM_ADDR_TAKE = 0X01
    SIM_CAM_PATH_LEN = 256
    CAM_SIZE_X = 1024
    CAM_SIZE_Y = 1024
