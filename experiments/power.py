import datetime
import pathlib
import sched
import subprocess
import threading
import time
import typing
from threading import Thread


class PowerLogger:
    def __init__(self, func, log_path):
        self.func = func
        self.log_path = log_path
        self.power_thread = PowerMeasurementThread(self.log_path)

    def set_current_document(self, current_document):
        self.power_thread.set_current_document(current_document)

    def set_current_fold_id(self, current_fold_id):
        self.power_thread.set_current_fold_id(current_fold_id)

    def start_logging(self):
        self.power_thread.start()
        self.func()
        self.power_thread.stop()


class PowerMeasurementThread(Thread):
    def __init__(self, log_path: str):
        super().__init__()
        self.event = None
        self.log_path = log_path
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.running = True
        self.current_document = None
        self.current_fold_id = None
        self.interval = 0.1

    def run(self):
        self.event = self.scheduler.enter(self.interval, 1, self.log_power_measurement)
        while self.running:
            self.scheduler.run(blocking=False)
            time.sleep(0)

    def stop(self):
        self.running = False
        if self.event is not None:
            self.scheduler.cancel(self.event)
        if threading.current_thread() != self:
            self.join()

    def set_current_document(self, current_document):
        self.current_document = current_document

    def set_current_fold_id(self, current_fold_id):
        self.current_fold_id = current_fold_id

    def log_power_measurement(self):
        completed_process = subprocess.run(['nvidia-smi', '--query-gpu=power.draw.instant', '--format=csv'],
                                           capture_output=True)
        process_output = completed_process.stdout
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = process_output.splitlines()[1].decode("utf-8")
        power = line.split(' ')[0]
        pathlib.Path(self.log_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.log_path, "a") as f:
            f.write(current_date)
            f.write("\t\t")
            if self.current_document is not None:
                f.write(str(self.current_document.id))
            else:
                f.write("None")
            f.write("\t\t")
            if self.current_fold_id is not None:
                f.write(str(self.current_fold_id))
            else:
                f.write("None")
            f.write("\t\t")
            f.write(power)
            f.write("\n")
        self.event = self.scheduler.enter(self.interval, 1, self.log_power_measurement)


if __name__ == "__main__":
    thread = PowerMeasurementThread("../res/power/test.dat")
    thread.start()
