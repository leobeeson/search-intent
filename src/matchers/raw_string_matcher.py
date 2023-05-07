

class RawStringMatcher:

    def __init__(self, query_index):
        self.query_index = query_index


    def match(self, text):
        return self.query_index.get(text, None)


    def add_query(self, text, category):
        self.query_index[text] = category
