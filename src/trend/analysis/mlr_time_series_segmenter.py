import multiprocessing
import signal
import piecewise_regression
import numpy as np

from trend.analysis.base_time_series_segmenter import BaseTimeSeriesSegmenter


class MlrTimeSeriesSegmenter(BaseTimeSeriesSegmenter):
    def __init__(self, min_segment_length: int = 4):
        super().__init__(min_segment_length)

    def segment(self, x, y: list[float] | list[int]) -> (list[tuple], list[dict]):
        x_copy, y_copy = x.copy(), y.copy()
        while y_copy[0] == 0 and y_copy[1] == 0:
            y_copy.pop(0)
            x_copy.pop(0)

        y_adjusted = (y_copy / np.max(y_copy)) * 100

        breakpoints = []
        segments = self.find_best_models(
            x_copy, y_adjusted, list(range(1, 11)), top_n=1, fit_repetitions=2, n_boot=500)

        if all(len(x) == 3 for x in segments):
            print("No breakpoints found")
        else:
            breakpoints = segments[0][2]

        if len(x_copy) != len(x) and x_copy[0] not in breakpoints:
            return [x_copy[0]] + breakpoints

        #  Unique list so we prevent duplicates
        return breakpoints

    def fit_model(self, x, y, n_breakpoints: int, fit_repetitions: int = 5, n_boot: int = 50) -> tuple:
        # https://github.com/tiangolo/fastapi/issues/1487
        # Needed to prevent FastAPI from shutting down
        signal.set_wakeup_fd(-1)

        min_score = 10**10
        best_results = None

        for _ in range(fit_repetitions):
            pw_fit = piecewise_regression.Fit(
                x, y, n_breakpoints=n_breakpoints, n_boot=n_boot, min_distance_between_breakpoints=2/len(x))
            results = pw_fit.get_results()
            if results["converged"] == False:
                break
            score = results["bic"] * results["rss"]
            if score < min_score:
                min_score = score
                best_results = results

        if best_results is None:
            return n_breakpoints, 10**10, None

        breakpoints = [round(best_results["estimates"]["breakpoint{}".format(i + 1)]["estimate"])
                       for i in range(n_breakpoints)]

        return n_breakpoints, min_score, breakpoints, best_results["bic"]

    def find_best_models(self, x, y, n_breakpoints: list[int], top_n: int = 4, fit_repetitions: int = 5, n_boot: int = 50, max_processes: int = 4) -> list[tuple]:
        with multiprocessing.Pool(processes=max_processes) as pool:
            results = pool.starmap(
                self.fit_model, [(x, y, n, fit_repetitions, n_boot) for n in n_breakpoints])

        return sorted(results, key=lambda x: x[1])[:top_n]
