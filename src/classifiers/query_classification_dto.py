
from dataclasses import dataclass


@dataclass
class QueryClassificationDTO:
    query: str
    raw_string_matcher: list[tuple[int, float]]
    preprocessed_string_matcher: list[tuple[int, float]]
    sequential_keyword_matcher: list[tuple[str, int, float]]
    bag_of_keyword_matcher: list[tuple[str, int, float]]

    def __init__(self, query: str):
        self.query = query
        self.raw_string_matcher = []
        self.preprocessed_string_matcher = []
        self.sequential_keyword_matcher = []
        self.bag_of_keyword_matcher = []

    def add_raw_match(self, category: int):
        self.raw_string_matcher.append((category, 1.0))

    def add_preprocessed_string_match(self, category: int):
        self.preprocessed_string_matcher.append((category, 1.0))

    def add_sequential_keyword_match(self, matches: list[tuple[str, int, float]]):
        self.sequential_keyword_matcher.extend(matches)

    def add_bag_of_keyword_match(self, matches: list[tuple[str, int, float]]):
        self.bag_of_keyword_matcher.extend(matches)