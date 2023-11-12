from models.models import Trend


class BaseTrendDescriptor:
    def generate_description(self, topics: list[str], start_year: int,
                             end_year: int, values: list[int],
                             global_trend: Trend, sub_trends: list[Trend]) -> str:
        raise NotImplementedError
