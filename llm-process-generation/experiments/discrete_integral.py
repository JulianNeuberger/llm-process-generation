import numpy as np
from datetime import datetime, timedelta
import statistics


# get timestamps and power measurements from logfile
def parse_logfile(file_path: str):
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
            for j in range(1, 20):
                # add last chars until a tab occurs
                if lines[i][-j] != "\t":
                    power_string = lines[i][-j] + power_string
                else:
                    break
            power_measurements.append(float(power_string))
    return timestamps, power_measurements


def calc_integral_trapezoid(timestamps: list[float], power_measurements: list[float]):
    microjoules = np.trapz(power_measurements, timestamps)
    joules = microjoules / 1000000
    kwh = round(joules / 3600000, 6)
    return kwh


def calc_integral_power_average(timestamps: list[float], power_measurements: list[float]):
    power_average = statistics.fmean(power_measurements)
    microjoules = power_average * timestamps[-1]
    joules = microjoules / 1000000
    kwh = round(joules / 3600000, 6)
    return kwh


def main():
    timestamps, power_measurements = parse_logfile("../res/power/ollama-llama3.3-70b-instruct 2025-01-25_17-06-20 pet_md-iterative1.dat")
    # conversion to seconds
    time_elapsed = round(timestamps[-1] / 1000000)
    average_power = round(statistics.fmean(power_measurements), 6)
    print("Experiment runtime: " + str(time_elapsed) + " seconds." + "\n" +
          "Average power draw: " + str(average_power) + " W" + "\n" +
          "Total energy used (trapezoid method): " + str(calc_integral_trapezoid(timestamps, power_measurements))
          + " kWh" + "\n" + "Total energy used (average power draw * runtime): " + str(
        calc_integral_power_average(timestamps, power_measurements)) + " kWh"
          )


if __name__ == "__main__":
    main()
