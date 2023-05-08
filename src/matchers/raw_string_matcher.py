

class RawStringMatcher:

    def __init__(self, query_index):
        self.query_index = query_index


    def match(self, query):
        return self.query_index.get(query, None)


    def add_query(self, query, category):
        self.query_index[query] = category


    def remove_query(self, query):
        if query in self.query_index:
            del self.query_index[query]


if __name__ == "__main__":
    raw_string_matcher = RawStringMatcher({"sample search query": 235})
    raw_string_matcher.match(None)