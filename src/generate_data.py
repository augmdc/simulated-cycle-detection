import simpy
import random
import numpy as np
import pandas as pd

# SIMULATION CONFIGURATION
NUM_LHD = 1
SIM_DURATION = 500 # Covers 1 full duty cycle
LOG_INTERVAL = 1 # Seconds between log entries

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
        # self.grade = 0

        # Start run process everytime an instance is created
        self.action = env.process(self.cycle())
        env.process(self.sensor_snapshot())

    def cycle(self):
        def randomize_activity_time(activity_time):
            return random.gauss(activity_time, activity_time * 0.1) # Not thread safe?

        while True:
            # TRANSIT
            self.state = "Transit"
            self.speed = MAX_SPEED_TRANSIT
            # self.grade = -13.0 # -13% grade
            transit_time = randomize_activity_time(60)
            yield self.env.timeout(transit_time)

            # LOAD
            self.state = "Load"
            self.speed = 4 # km/h
            load_time = randomize_activity_time(30)
            yield self.env.timeout(load_time)

            # Haul
            self.state = "Haul"
            self.speed = 8
            haul_time = randomize_activity_time(90)
            yield self.env.timeout(haul_time)

            # Dump
            self.state = "Dump"
            self.speed = 4
            dump_time = randomize_activity_time(20)
            yield self.env.timeout(dump_time)

            # END OF THE CYCLE

    def sensor_snapshot(self):
        """
        Collects sensor data every 1 second.
        """
        while True:
            # Add Gaussian Noise to simulate real sensors
            noise_speed = self.speed + random.normalvariate(0, 1.5)
            
            # Ensure physics constraints (no negative speed/payload)
            noise_speed = max(0, noise_speed)
            telemetry.append({
                'timestamp': self.env.now,
                'speed': noise_speed,
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

