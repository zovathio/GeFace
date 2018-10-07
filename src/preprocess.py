import numpy as np
import pandas as pd
import os
import sys

src_folder = os.path.dirname(__file__)

csv_indexer = {"dob": 0, "photo_taken":1, "full_path":2}


#os.chdir("../data")
try:
    csv_file = pd.read_csv("../data/imdb.csv", ",")
    header = pd.read_csv("../data/imdb.csv", ",", header=None, nrows=1)
    pd.set_option('display.max_columns', 100)
    #print(csv_file.head(1))

#   print(type(csv_file))
    matrix = csv_file.as_matrix()

    #print(matrix[1:])
    new_data = [csv_file.get("photo_taken"), csv_file.get("gender"), csv_file.get("full_path")]
    # print(csv_file.get("full_path"))
    # print(csv_file.get("gender"))
    # print(csv_file.get("photo_taken"))
    for data in new_data:
        print(data[0], data[1], data[2])

except (FileNotFoundError):
    print("CSV file not found")
