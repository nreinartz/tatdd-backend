from __future__ import annotations

from enum import Enum
from models.models import Trend, TrendType
from trend.descriptor.base_descriptor import BaseTrendDescriptor
import random
import numpy as np


class TrendInterest(Enum):
    NONE = 0,
    VERY_LOW = 1,
    LOW = 2,
    MODERATE = 3,
    HIGH = 4,
    VERY_HIGH = 5


class TrendLength(Enum):
    SHORT = 0,
    MEDIUM = 1,
    LONG = 2


def get_rule_based_descriptor():
    return RuleBasedDescriptor()


descriptions = {
    TrendType.NONE: {
        TrendLength.SHORT: [
            " a short period of stability was observed. ",
            " the trend remained flat for a short duration. ",
            " the trend showed a short period of steadiness. ",
            " the trend exhibited a short period of equilibrium. ",
            " there was a short steady state for the topic."
        ],
        TrendLength.MEDIUM: [
            " the trend consistently showed stability.",
            " there was a steady state in the trend.",
            " the trend exhibited neutrality.",
            " an unvarying trend was noticed.",
            " the trend sustained its equilibrium."
        ],
        TrendLength.LONG: [
            " a long lack of variation characterized the trend.",
            " the trend demonstrated steadiness over a long period.",
            " there was a long-term stability in the trend.",
            " the trend maintained an even keel over a long duration.",
            " there was an enduring consistency in the trend."
        ]
    },
    TrendType.INCREASING: {
        TrendLength.SHORT: [
            " a {adjective} uptick was noted in the trend.",
            " the trend took a {adjective} upward turn.",
            " there was a {adjective} increase in the trend.",
            " the trend experienced a {adjective} rise.",
            " a {adjective} surge was observed in the trend."
        ],
        TrendLength.MEDIUM: [
            " the trend witnessed a {adjective} upward movement.",
            " there was a {adjective} mid-term rise in the trend.",
            " the trend demonstrated a {adjective} ascent.",
            " the trend showed a {adjective} positive swing.",
            " an {adjective} upward shift was observed in the trend."
        ],
        TrendLength.LONG: [
            " a {adjective} long-term uptrend was evident.",
            " the trend indicated a {adjective} sustained rise.",
            " there was a {adjective} considerable upswing in the trend over a long period.",
            " the trend exhibited a {adjective} prolonged upward movement.",
            " a {adjective} lasting rise characterized the trend."
        ]
    },
    TrendType.DECREASING: {
        TrendLength.SHORT: [
            " a {adjective} downturn was observed in the trend.",
            " the trend took a {adjective} downward dive.",
            " there was a {adjective} decrease in the trend.",
            " a {adjective} dip was noticed in the trend.",
            " the trend showed a {adjective} decline.",
            " a {adjective} drop was noted in the trend.",
            " the trend experienced a {adjective} fall."
        ],
        TrendLength.MEDIUM: [
            " the trend experienced a {adjective} downward movement.",
            " there was a {adjective} mid-term fall in the trend.",
            " the trend demonstrated a {adjective} descent.",
            " a {adjective} negative swing was observed in the trend.",
            " the trend went through a {adjective} downward shift.",
            " the trend showed a {adjective} negative trend."
        ],
        TrendLength.LONG: [
            " a {adjective} long-term downtrend was apparent.",
            " the trend indicated a {adjective} sustained decline.",
            " there was a {adjective} significant downslide in the trend over a long period.",
            " the trend exhibited a {adjective} prolonged downward movement.",
            " a {adjective} lasting fall characterized the trend.",
            " the trend showed a {adjective} negative trend over a long period."
        ]
    }
}

# Define trend change transitions
transitions = {
    (TrendType.NONE, TrendType.INCREASING): [
        "Transitioning from a period of stability, a rising trend began to form.",
        "Shifting from neutrality, an uptrend started to emerge.",
        "From a state of no trend, the graph started to climb, indicating an uptrend.",
        "Breaking the steady pattern, a new uptrend was observed.",
        "Moving away from the flat trend, the curve began to rise."
    ],
    (TrendType.NONE, TrendType.DECREASING): [
        "From a flat trend, a downturn started to develop.",
        "Leaving the stability behind, the trend started to dip.",
        "Departing from a period of no trend, a falling trend took shape.",
        "Breaking the steadiness, a new downtrend was noticed.",
        "Transitioning from the neutral state, the graph started to fall."
    ],
    (TrendType.INCREASING, TrendType.NONE): [
        "A previous uptrend began to stabilize into a period of no significant change.",
        "Moving from an ascending trend, the graph started to flatten.",
        "Transitioning from an uptrend, a steady pattern started to form.",
        "The rising trend started to level off, indicating a period of no trend.",
        "From an uptrend, the graph started to show a neutral trend."
    ],
    (TrendType.INCREASING, TrendType.DECREASING): [
        "A rising trend started to reverse, transitioning into a downtrend.",
        "From an uptrend, a falling trend started to develop.",
        "The graph began to dip, indicating a shift from an uptrend to a downtrend.",
        "Moving away from the uptrend, a downtrend started to take shape.",
        "The rising trend started to fall, marking the beginning of a downturn."
    ],
    (TrendType.DECREASING, TrendType.INCREASING): [
        "A falling trend began to reverse, showing signs of an uptrend.",
        "From a downtrend, the graph started to climb.",
        "Moving away from the downward slope, an uptrend started to form.",
        "The falling trend started to rise, indicating a shift to an uptrend.",
        "The downward trend reversed, and an upward trend started to emerge."
    ],
    (TrendType.DECREASING, TrendType.NONE): [
        "A falling trend began to stabilize, indicating a period of no trend.",
        "From a downtrend, the graph started to flatten.",
        "The downward trend started to show signs of steadiness.",
        "Moving away from the downtrend, a neutral trend started to form.",
        "A downtrend began to level off, transitioning into a period of no significant change."
    ],
}

time_phrases = [
    "Towards the close of {start} and the beginning of {end}, ",
    "In the timespan spreading from {start} to {end}, ",
    "Throughout the stretch from {start} to {end}, ",
    "In the period commencing {start}, stretching till {end}, ",
    "Between the years {start} and {end}, ",
    "From {start}, continuing through to {end}, ",
    "In the years following {start} up until {end}, ",
    "Traversing the timeline from {start} to {end}, ",
    "In the interval starting {start} and ending {end}, "
]

time_phrases_continuous = [
    "After that, until {end}, ",
    "From that point on, until {end}, ",
    "Continuing from year {start} to {end}, ",
    "From {start} onward, ",
]

combined_time_phrases = time_phrases + time_phrases_continuous

value_description_prefixes = [
    "During that time, ",
    "In that period, ",
    "In that interval, ",
    "Throughout that stretch, ",
    "Over that duration, ",
    "In that span, ",
    "In that time frame, ",
    "In that time period, ",
    "Between these years, "
]

value_descriptions = {
    TrendInterest.NONE: [
        "there was a complete lack of interest in the topic.",
        "the interest in the topic was practically nonexistent."
    ],
    TrendInterest.VERY_LOW: [
        "there was very little interest in the topic.",
        "the topic attracted minimal attention."
    ],
    TrendInterest.LOW: [
        "interest in the topic was low.",
        "the topic had limited appeal."
    ],
    TrendInterest.MODERATE: [
        "the topic garnered moderate interest.",
        "there was a fair amount of attention to the topic."
    ],
    TrendInterest.HIGH: [
        "the topic was quite popular.",
        "there was significant interest in the topic."
    ],
    TrendInterest.VERY_HIGH: [
        "the topic was extremely popular.",
        "interest in the topic was at its peak."
    ]
}


class Segment:
    def __init__(self, start_year, end_year, trend_type: TrendType, slope, values: list[int]):
        self.start_year = start_year
        self.end_year = end_year
        self.trend_type = trend_type
        self.slope = slope

        self.interest = self.get_interest(values)
        self.length = self.get_length()

    def get_adjective(self):
        if abs(self.slope) < 0.2:
            return "slight"
        elif abs(self.slope) < 0.5:
            return "moderate"
        else:
            return "steep"

    def get_interest(self, values):
        average = np.average(values)

        if average < 2:
            return TrendInterest.NONE
        elif average < 10:
            return TrendInterest.VERY_LOW
        elif average < 20:
            return TrendInterest.LOW
        elif average < 40:
            return TrendInterest.MODERATE
        elif average < 70:
            return TrendInterest.HIGH
        else:
            return TrendInterest.VERY_HIGH

    def get_length(self):
        length = self.end_year - self.start_year
        return TrendLength.LONG if length > 5 else TrendLength.MEDIUM if length > 1 else TrendLength.SHORT

    def describe(self, previous: Segment | None = None):
        transition = ""
        if previous:
            if previous.trend_type != self.trend_type and transitions and random.random() < 0.4:
                transition = random.choice(
                    transitions[(previous.trend_type, self.trend_type)])
                transition = " {} ".format(transition)

        # Transition
        description = transition

        # Time description
        description += random.choice(time_phrases if previous == None else combined_time_phrases).format(
            start=self.start_year, end=self.end_year)

        # Main description
        description += random.choice(descriptions[self.trend_type][self.length]).format(
            adjective=self.get_adjective())

        value_description = ""
        if random.random() < 0.4:
            value_description = random.choice(value_description_prefixes) + \
                random.choice(value_descriptions[self.interest])

        description += " {} ".format(value_description)

        return description


class RuleBasedDescriptor(BaseTrendDescriptor):
    def generate_description(self, topics: list[str], start_year: int,
                             end_year: int, values: list[int],
                             global_trend: Trend, sub_trends: list[Trend]) -> str:

        if len(sub_trends) == 0:
            return "No trends were detected."

        segments = [Segment(t.start, t.end, t.type, t.slope, values[t.start - start_year:t.end - start_year + 1])
                    for t in sub_trends]
        descriptions = [segments[0].describe()]
        for i in range(1, len(segments)):
            descriptions.append(segments[i].describe(segments[i-1]))
        return " ".join(descriptions)
