import datetime
import pathlib
import sched
import subprocess
import threading
import time
from threading import Thread, Lock


class PowerLogger:
    def __init__(self, func, log_path, args=None):
        self.func = func
        self.log_path = log_path
        self.power_thread = PowerMeasurementThread(self.log_path)
        self.args = args if args is not None else []

    def set_current_document(self, current_document):
        self.power_thread.set_current_document(current_document)

    def set_current_fold_id(self, current_fold_id):
        self.power_thread.set_current_fold_id(current_fold_id)

    def start_logging(self):
        self.power_thread.start()
        self.func(*self.args)
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
        self.lock = Lock()  # Protects self.event

    def run(self):
        with self.lock:
            self.event = self.scheduler.enter(self.interval, 1, self.log_power_measurement)
        while self.running:
            self.scheduler.run(blocking=False)
            time.sleep(0)

    def stop(self):
        self.running = False
        with self.lock:
            if self.event:
                try:
                    self.scheduler.cancel(self.event)
                except ValueError:
                    pass
                self.event = None

        if threading.current_thread() != self:
            self.join()

    def set_current_document(self, current_document):
        self.current_document = current_document

    def set_current_fold_id(self, current_fold_id):
        self.current_fold_id = current_fold_id

    def log_power_measurement(self):
        if not self.running:
            return

        completed_process = subprocess.run(['nvidia-smi', '--query-gpu=power.draw.instant,memory.used', '--format=csv'],
                                           capture_output=True)
        process_output = completed_process.stdout.decode("utf-8")
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        lines = process_output.splitlines()
        power = lines[1].split(' ')[0] if len(lines) > 1 else "error"
        memory = lines[1].split(' ')[1] if len(lines) > 1 else "error"
        pathlib.Path(self.log_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.log_path, "a") as f:
            f.write(f"{current_date}\t\t{self.current_document.id if self.current_document else 'None'}\t\t"
                    f"{self.current_fold_id if self.current_fold_id else 'None'}\t\t{power}\t{memory}\n")

        with self.lock:
            if self.running:
                self.event = self.scheduler.enter(self.interval, 1, self.log_power_measurement)


if __name__ == "__main__":
    thread = PowerMeasurementThread("../res/power/test.dat")
    thread.start()
