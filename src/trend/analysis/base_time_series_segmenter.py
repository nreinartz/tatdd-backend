class BaseTimeSeriesSegmenter:
    def __init__(self, min_segment_length: int = 4):
        self.min_segment_length = min_segment_length

    def segment(self, x, y) -> list[int]:
        pass
