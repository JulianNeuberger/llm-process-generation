import pathlib
import sched
import subprocess
import time
import typing
from threading import Thread


def measure_power_draw_for_function(func, log_path):
    power_thread = PowerMeasurementThread(log_path)
    power_thread.start()
    func()
    power_thread.stop()

class PowerLogger:
    def __init__(self):
        self.current_document = None
        self.current_fold_id = None

    def measure_power_draw_for_function(self, func, log_path):
        power_thread = PowerMeasurementThread(log_path)
        power_thread.start()
        func()
        power_thread.stop()

class PowerMeasurementThread(Thread):
    def __init__(self, log_path: str):
        super().__init__()
        self.event = None
        self.log_path = log_path
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.running = True
        self.current_document = None
        self.current_fold_id = None

    def run(self):
        self.event = self.scheduler.enter(1, 1, self.log_power_measurement)
        while self.running:
            self.scheduler.run(blocking=False)
            time.sleep(0)

    def stop(self):
        self.running = False
        if self.event:
            self.scheduler.cancel(self.event)

    def log_power_measurement(self):
        completed_process = subprocess.run(['nvidia-smi', '--query-gpu=power.draw.average', '--format=csv'],
                                           capture_output=True)
        process_output = completed_process.stdout
        line = process_output.splitlines()[1].decode("utf-8")
        power = line.split(' ')[0]
        pathlib.Path(self.log_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.log_path, "a") as f:
            if self.current_document is not None:
                f.write(self.current_document.id)
            else:
                f.write("None")
            f.write("\t")
            if self.current_fold_id is not None:
                f.write(self.current_fold_id)
            else:
                f.write("None")
            f.write("\t")
            f.write(power)
            f.write("\n")
        self.event = self.scheduler.enter(1, 1, self.log_power_measurement)


if __name__ == "__main__":
    thread = PowerMeasurementThread("../res/power/test.dat")
    thread.start()
