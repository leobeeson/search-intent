from src.matchers.raw_string_matcher import RawStringMatcher


class ClassificationPipeline:
    
    
    def __init__(self, query_index):
        self.exact_raw_string_matcher = RawStringMatcher(query_index)
        

    def classify(self, query: str):
        result = self.exact_raw_string_matcher.match(query)
        if result is not None:
            return result

        return None
