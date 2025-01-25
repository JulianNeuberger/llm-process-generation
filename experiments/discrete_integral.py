import numpy as np
from datetime import datetime, timedelta


# get timestamps and power measurements from logfile
def parse(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # timestamps are 26 characters long
        initial_timestamp = datetime.strptime(lines[0][0:26], "%Y-%m-%d %H:%M:%S.%f")
        timestamps = [0.0]
        for i in range(1, len(lines)):
            # calc time elapsed since the start
            delta = datetime.strptime(lines[i][0:26], "%Y-%m-%d %H:%M:%S.%f") - initial_timestamp
            # convert to microseconds
            delta_micro = delta / timedelta(microseconds=1)
            timestamps.append(delta_micro)

        power_measurements = []
        for i in range(len(lines)):
            power_string = ""
            # power can be a max of 6 chars long (456.78)
            for j in range(1, 6):
                # add last chars until a tab occurs
                if lines[i][-j] != "\t":
                    power_string = lines[i][-j] + power_string
                else:
                    break
            power_measurements.append(power_string)
    return timestamps, power_measurements

