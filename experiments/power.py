import os
import pathlib
import sched, time
import subprocess
from threading import Thread


def measure_power_draw_for_function(func, log_path):
    thread = PowerMeasurementThread(log_path)
    thread.start()
    func()
    thread.stop()


class PowerMeasurementThread(Thread):
    def __init__(self, log_path: str):
        super().__init__()
        self.event = None
        self.log_path = log_path
        self.scheduler = sched.scheduler(time.time, time.sleep)

    def run(self):
        self.event = self.scheduler.enter(1, 1, self.log_power_measurement, (self.scheduler,))
        self.scheduler.run()

    def stop(self):
        self.scheduler.cancel(self.event)

    def log_power_measurement(self, scheduler):
        completed_process = subprocess.run(['nvidia-smi', '--query-gpu=power.draw.average', '--format=csv'],
                                           capture_output=True)
        print(completed_process.returncode)
        process_output = completed_process.stdout
        line = process_output.splitlines()[1].decode("utf-8")
        power = line.split(' ')[0]
        pathlib.Path(self.log_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.log_path, "a") as f:
            f.write(power)
            f.write("\n")
        self.event = scheduler.enter(1, 1, self.log_power_measurement, (scheduler,))
        scheduler.run()


if __name__ == "__main__":
    thread = PowerMeasurementThread("../res/power/test.dat")
    thread.start()
