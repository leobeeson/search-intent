from src.matchers.raw_string_matcher import RawStringMatcher


class ClassificationPipeline:
    
    
    def __init__(self, training_data: dict[str, int]):
        self.exact_raw_string_matcher: RawStringMatcher = RawStringMatcher(training_data)
        

    def classify(self, query: str):
        category = self.exact_raw_string_matcher.match(query)
        if category >= 0:
            return category
        return category
