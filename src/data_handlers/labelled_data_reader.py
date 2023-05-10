import csv


class LabelledDataReader:
    
    
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.labelled_data: dict = {}

    
    def read_data(self):
        with open(self.filepath, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.labelled_data[row[0]] = int(row[1])
        return self.labelled_data
