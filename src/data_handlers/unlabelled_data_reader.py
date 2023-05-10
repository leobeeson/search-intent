import csv
import os


class UnlabelledDataReader:

    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.unlabelled_data: dict[str, int] = {}


    def read_data(self):
        _, file_extension = os.path.splitext(self.filepath)
        if file_extension == ".csv":
            self._read_csv()
        elif file_extension == ".txt":
            self._read_txt()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions are .csv and .txt.")
        return self.unlabelled_data


    def _read_csv(self):
        with open(self.filepath, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.unlabelled_data[row[0]] = None


    def _read_txt(self):
        with open(self.filepath, "r") as txt_file:
            for line in txt_file:
                self.unlabelled_data[line.rstrip()] = None