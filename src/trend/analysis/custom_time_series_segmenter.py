import math
import numpy as np
from operator import itemgetter
from base_time_series_segmenter import BaseTimeSeriesSegmenter
from scipy.signal import sosfilt, butter


class CustomTimeSeriesSegmenter(BaseTimeSeriesSegmenter):
    def __init__(self, min_segment_length: int = 4, divide_iterations: int = 4, conquer_cutoff_angle: float = 120):
        super().__init__(min_segment_length)
        self.divide_iterations = divide_iterations
        self.conquer_cutoff_angle = conquer_cutoff_angle
        self.sos = butter(1, .3, btype='low', analog=False, output='sos')

    def segment(self, x, y: list[float] | list[int]) -> (list[tuple], list[dict]):
        y_adjusted = (y / np.max(y)) * 100
        smoothed_y = list(sosfilt(self.sos, y_adjusted))

        # Divide and conquer
        divide_segments = self.__divide(x, smoothed_y, self.divide_iterations)
        conquer_segments = self.__conquer(
            x, smoothed_y, divide_segments, self.conquer_cutoff_angle)

        return conquer_segments, self.__get_segmentation_from_segments(x, smoothed_y, conquer_segments)

    def __divide(self, x, y, iterations: int = 4, min_width: int = 2):
        if len(x) < min_width:
            return []

        segments = [(0, len(x) - 1)]
        for _ in range(iterations):
            segments = sum([self.__split_segment(x, y, *segment, min_width)
                           for segment in segments], ())
        return segments

    def __conquer(self, x, y, segments, cutoff_angle: float = 120):
        sgmts = list(segments)
        angles = [self.__measure_angle(
            x, y, segments[i], segments[i + 1]) for i in range(len(segments) - 1)]
        eligible_segments = sorted([(angles[i], segments[i]) for i in range(
            len(angles)) if angles[i] > cutoff_angle], key=itemgetter(0), reverse=True)
        print(eligible_segments)

        for _, segment in eligible_segments:
            if segment not in sgmts:
                continue
            sgmt_i = sgmts.index(segment)
            sgmts = sgmts[:sgmt_i] + [(segments[sgmt_i][0],
                                       segments[sgmt_i + 1][1])] + sgmts[sgmt_i + 1:]
        return sgmts

    def __split_segment(self, x, y, start, end, min_segment_size: int = 3) -> int | None:
        current_deviation = self.__measure_deviation(x, y, start, end)
        current_center = -1

        if start + min_segment_size >= end - min_segment_size:
            return ((start, end),)

        for center in range(start + min_segment_size, end - min_segment_size):
            left_deviation = self.__measure_deviation(x, y, start, center)
            right_deviation = self.__measure_deviation(x, y, center, end)
            if left_deviation + right_deviation < current_deviation:
                current_deviation = left_deviation + right_deviation
                current_center = center

        if current_center == -1:
            return ((start, end),)

        return ((start, current_center), (current_center, end))

    def __measure_angle(self, x, y, firstSegment, secondSegment):
        u = np.array([x[firstSegment[1]] - x[firstSegment[0]],
                     y[firstSegment[1]] - y[firstSegment[0]]])
        v = np.array([x[secondSegment[1]] - x[secondSegment[0]],
                     y[secondSegment[1]] - y[secondSegment[0]]])

        angle = math.acos(
            np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return 180 - math.degrees(angle)

    def __get_slope(self, x, y, segment):
        return (y[segment[1]] - y[segment[0]]) / (x[segment[1]] - x[segment[0]])

    def __measure_deviation(self, x, y, start, end):
        linear_eq = np.poly1d(np.polyfit(x[start:end], y[start:end], 1))
        return np.sum(np.abs(y[start:end] - linear_eq(x[start:end])))

    def __get_segmentation_from_segments(self, x, y, segments, min_width: int = 4):
        splits = []
        for i in range(len(segments) - 1):
            (leftStart, leftEnd) = segments[i]
            leftStart = 0 if len(splits) == 0 else splits[-1]

            if leftEnd - leftStart < min_width:
                continue

            # For angle measurement we take the correct segment
            angle = self.__measure_angle(x, y, segments[i], segments[i + 1])

            left_slope = self.__get_slope(x, y, segments[i])
            right_slope = self.__get_slope(x, y, segments[i + 1])
            slope_signs = (np.sign(left_slope), np.sign(right_slope))

            lSlope = np.log(1 + left_slope**2)
            rSlope = np.log(1 + right_slope**2)

            diff = np.abs(lSlope - rSlope)
            print(diff)
            print(angle)
            if diff > 2.3 or (np.sum(list(slope_signs)) == 0 and angle < 80):
                splits.append(leftEnd)

        return splits
