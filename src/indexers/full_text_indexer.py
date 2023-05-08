import csv


class FullTextIndexer:
    
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.query_index = {}

    
    def index_data(self):
        with open(self.filepath, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.query_index[row[0]] = int(row[1])
        return self.query_index
