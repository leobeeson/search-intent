from src.matchers.raw_string_matcher import RawStringMatcher
from src.classifiers.query_classification_dto import QueryClassificationDTO


class ClassificationPipeline:
    
    
    def __init__(self, training_data: dict[str, int]):
        self.classification_dto: QueryClassificationDTO = None
        self.exact_raw_string_matcher: RawStringMatcher = RawStringMatcher(training_data)
        

    def classify(self, query: str):
        self.classification_dto: QueryClassificationDTO = QueryClassificationDTO(query)
        
        category = self.exact_raw_string_matcher.match(query)
        if category >= 0:
            self.classification_dto.add_raw_match(category)
            return category
        
        return category
