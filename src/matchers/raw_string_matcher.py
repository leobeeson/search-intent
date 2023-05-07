

class RawStringMatcher:

    def __init__(self, query_index):
        self.query_index = query_index

    def match(self, text):
        return self.query_index.get(text, None)
