import numpy as np
from datetime import datetime

# get timestamps and power measurements from logfile
def parse(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # timestamps are 23 characters long
        initial_timestamp = datetime.strptime(lines[0][0:23], "%Y-%m-%d %H:%M:%S.%f")
        timestamps = [initial_timestamp]
        for i in range(1, len(lines)):
            timestamps.append(lines[i][0:23])

        power_measurements = []
    return timestamps, power_measurements

