from itertools import groupby
import pymannkendall as mk
import numpy as np

from trend.analysis.mlr_time_series_segmenter import MlrTimeSeriesSegmenter
from models.models import Trend, TrendType


def get_trend_analyser():
    return TrendAnalyser()


class TrendAnalyser:
    def analyse(self, x, y) -> (list[int], list[Trend]):
        time_series_segmenter = MlrTimeSeriesSegmenter(min_segment_length=4)
        cuts = time_series_segmenter.segment(x, y)
        cuts_i = [0] + [x.index(cut) for cut in cuts] + [len(x) - 1]

        # Sub-trends
        segments = [(cuts_i[i], cut) for i, cut in enumerate(cuts_i[1:])]

        y_adjusted = (y / np.max(y)) * 100

        trends = self.__get_trends_for_segments(x, y_adjusted, segments)
        trend_slopes = [trends[i].type.value for i in range(len(trends))]

        merged_segments = []
        for k, g in groupby(enumerate(trend_slopes), key=lambda i_x: i_x[1]):
            group = list(g)
            start_index = group[0][0]
            end_index = group[-1][0]
            merged_segments.append(
                (segments[start_index][0], segments[end_index][1]))

        # Whole time series
        merged_segments = [(0, len(x) - 1)] + merged_segments

        return [x[start] for (start, _) in merged_segments[2:]], self.__get_trends_for_segments(x, y_adjusted, merged_segments)

    def __get_trends_for_segments(self, x, y, segments):
        slopes = [mk.sens_slope(
            y[start:end + 1]).slope for start, end in segments]
        trends = [0 if np.abs(slope) < 1 else int(np.sign(slope))
                  for slope in slopes]

        return [
            Trend(
                start=x[segments[i][0]],
                end=x[segments[i][1]],
                type=TrendType.NONE if trends[i] == 0 else (
                    TrendType.INCREASING if trends[i] == 1 else TrendType.DECREASING),
                slope=slopes[i],
                line=np.polyfit(x[segments[i][0]:segments[i][1] + 1],
                                y[segments[i][0]:segments[i][1] + 1], 1).tolist()
            ) for i in range(len(segments))
        ]

    # pettit test
