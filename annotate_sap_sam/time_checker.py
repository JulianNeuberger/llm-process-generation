from typing import Callable
from typing_extensions import ParamSpec, TypeVar
import time
import numpy as np
import matplotlib.pyplot as plt

F_Spec = ParamSpec("F_Spec")       # Parameters of the wrapped function
F_Return = TypeVar("F_Return")         # Return type of the wrapped function

class TimeChecker:

    func: Callable[F_Spec, F_Return]
    timings: list[float]

    def __init__(self, func: Callable[F_Spec, F_Return]):
        self.func = func
        self.timings = list()

    def __call__(self, *args: F_Spec.args, **kwargs: F_Spec.kwargs) -> F_Return:
        start_time = time.time()
        result = self.func(*args, **kwargs)
        duration = time.time() - start_time

        self.timings.append(duration)

        return result

    def report(self) -> None:
        count = len(self.timings)
        if count == 0:
            print(f"{self.func.__name__}: no recorded calls.")
            return

        avg = np.mean(self.timings)
        std = np.std(self.timings)
        total = sum(self.timings)

        print(f"{self.func.__name__}:")
        print(f"  Calls: {count}")
        print(f"  Avg:   {avg:.4f}s")
        print(f"  Std:   {std:.4f}s")
        print(f"  Total: {total:.4f}s")

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(self.timings, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(avg, color='red', linestyle='--', label=f'Mean: {avg:.4f}s')
        plt.axvline(avg + std, color='orange', linestyle='--', label=f'+1σ: {avg + std:.4f}s')
        plt.axvline(avg - std, color='orange', linestyle='--', label=f'-1σ: {avg - std:.4f}s')
        plt.title(f"Execution Time Distribution: {self.func.__name__}")
        plt.xlabel("Execution Time (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # save timings to a file
        with open(f"{self.func.__name__}_timings.txt", "w") as f:
            for timing in self.timings:
                f.write(f"{timing:.6f}\n")
