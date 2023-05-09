import csv


class LabelledDataSet:
    
    
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.query_index: dict = {}

    
    def index_data(self):
        with open(self.filepath, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.query_index[row[0]] = int(row[1])
        return self.query_index
