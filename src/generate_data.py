import simpy
import random
import numpy as np
import pandas as pd

# SIMULATION CONFIGURATION
NUM_LHD = 1
# CYCLE_DURATION = 200 # seconds
# CYCLES = 1
SIM_DURATION = 500
LOG_INTERVAL = 1 # Seconds between log entries

# CYCLE CONFIG
TRANSIT_TIME = 60
LOAD_TIME = 30
HAUL_TIME = 90
DUMP_TIME = 20
TIME_VARIANCE = 0.1  # 10% standard deviation

ACTIVITY_CONFIG = {
    'Transit': (60, 0.1),   # Consistent route
    'Load':    (30, 0.3),   # Highly variable (material conditions)
    'Haul':    (90, 0.15),  # Some traffic variation
    'Dump':    (20, 0.1),   # Fairly consistent
}

# PHYSICS CONSTANTS
MAX_SPEED_TRANSIT = 12 #km/h
MAX_SPEED_HAUL = 8 # km/h

# Global list to store sensor data
telemetry = []

class LHD:
    def __init__(self, env):
        self.env = env

        # Set initial conditions of the vehicle
        self.state = "Idle"
        self.speed = 0
        # self.direction = 0
        self.grade = 0 # Start on flat ground

        # Start run process everytime an instance is created
        self.action = env.process(self.cycle())
        env.process(self.sensor_snapshot())

    def cycle(self):
        def randomize_activity_time(activity_time, variance=0.1):
            return random.gauss(activity_time, activity_time * variance) # Not thread safe?

        while True:
            # TRANSIT
            self.state = "Transit"
            self.speed = MAX_SPEED_TRANSIT
            self.grade = -13.0 # -13% going downhill
            transit_time = randomize_activity_time(ACTIVITY_CONFIG["Transit"][0], ACTIVITY_CONFIG["Transit"][0])
            yield self.env.timeout(transit_time)

            # TO-DO: have grade change while evening out at bottom of the hill

            # Load
            # TO-DO: Speed starts dropping off, approaches near zero at end of Load
            self.state = "Load"
            self.speed = 4 # km/h
            self.grade = 0
            load_time = randomize_activity_time(
                ACTIVITY_CONFIG["Load"][0],
                ACTIVITY_CONFIG["Load"][1]
                )
            yield self.env.timeout(load_time)

            # Haul
            self.state = "Haul"
            self.speed = 8
            self.grade = 13.0 # 13% (going uphill)
            haul_time = randomize_activity_time(
                ACTIVITY_CONFIG["Haul"][0],
                ACTIVITY_CONFIG["Haul"][1]
                )
            yield self.env.timeout(haul_time)

            # TO-DO: have grade change while cresting the hill

            # Dump
            # TO-DO: Speed zigzags up and down
            self.state = "Dump"
            self.speed = 4
            self.grade = 0
            dump_time = randomize_activity_time(
                ACTIVITY_CONFIG["Dump"][0],
                ACTIVITY_CONFIG["Dump"][1]
                )
            yield self.env.timeout(dump_time)

            # END OF THE CYCLE

    def sensor_snapshot(self):
        """
        Collects sensor data every 1 second.
        """
        while True:
            # Add Gaussian Noise to simulate real sensors
            noise_speed = self.speed + random.normalvariate(0, 1.5)
            noise_grade = self.grade + random.normalvariate(0, 1.5)
            
            # Ensure physics constraints (no negative speed/payload)
            noise_speed = max(0, noise_speed)
            telemetry.append({
                'timestamp': self.env.now,
                'speed': noise_speed,
                'grade': noise_grade,
                'state': self.state # This is the LABEL for your ML model
            })
            yield self.env.timeout(LOG_INTERVAL)

# RUN SIMULATION
env = simpy.Environment()
for i in range(NUM_LHD):
    LHD(env)

env.run(until=SIM_DURATION)

# Export telemetry data
df = pd.DataFrame(telemetry)
df.to_csv('data/telemetry.csv', index=False)
print(f"Simulation complete. Generated {len(telemetry)} telemetry records.")