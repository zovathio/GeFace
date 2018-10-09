import numpy as np
import pandas as pd
import os
import sys
import random
import re
import datetime
from matplotlib import pyplot as plt
import  collections

random.seed(42)
np.random.seed(42)

src_folder = os.path.dirname(__file__)

csv_indexer = {"dob": 0, "photo_taken":1, "full_path":2}
train_data = []
valid_data = []
test_data = []
date_of_birth = []
age = []

try:
    # read the csv
    csv_file = pd.read_csv("../data/imdb.csv", ",")
    header = pd.read_csv("../data/imdb.csv", ",", header=None, nrows=1)
    pd.set_option('display.max_columns', 100)

    full_path = csv_file.get("full_path")
    #print(len(full_path))
    i = 0;
    bad = []
    for elem in full_path:
        m = re.search("[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}", elem)
        if m is None:
            # if something is not okay with the date than skipp and mark the index
            bad.append(i)
            continue
        mm = re.search("[0-9]{4}", m.group(0))
        date_of_birth.append(int(mm.group(0)))
        i += 1

    # print("BAD:")
    # print(bad)
    # print("Bad len: ", len(bad), "\n")

    # get the neccessary columns from pandas and create a numpy array
    new_data = np.asmatrix([
                          csv_file.get("photo_taken"),
                          csv_file.get("gender"),
                          csv_file.get("full_path")],
    )

    # transpose the matrix
    new_data = np.transpose(new_data)
    # remove the wrong entries
    for elem in bad:
        new_data = np.delete(new_data, elem, axis=0)

    new_data = np.insert(new_data, 0, np.asarray(date_of_birth), axis=1)
    # claculate ages for the persons
    for i in range(new_data.shape[0]):
        age.append(new_data[i, 1] - new_data[i, 0])
    # insert ages into the 3 column
    new_data = np.insert(new_data, 2, np.asarray(age), axis=1)

    # new_data = np.transpose(new_data)
    print(new_data)
    print(new_data.shape)
    print()

    # shuffle the data by rows
    np.random.shuffle(new_data)
    print("shuffled data: ")
    print(new_data)

    # calculate the numbers of datas
    test_num = np.floor(0.1 * new_data.shape[0])
    valid_num = np.floor(0.2 * new_data.shape[0])
    train_num = new_data.shape[0] - test_num - valid_num

    print("\nTrain data number: {0} | valid data number: {1} | test data number: {2}\n".format( train_num, valid_num, test_num))

    # split the data
    train_data = new_data[:, 0:int(train_num)]
    valid_data = new_data[:, 0:int(valid_num)]
    test_data = new_data[:, 0:int(test_num)]
    # print(train_data)


    bad = []
    # create array for histogram
    hist = np.zeros(101)
    x = range(101)
    # mark bad elemnt index
    print(new_data.shape)
    for i in range(new_data.shape[0]):
        if new_data[i, 2] < 0 or new_data[i, 2] > 100:
            bad.append(i)

    print(bad)
    print(len(bad))
    print(new_data.shape)
    # remove bad elements
    for elem in bad:
        # print(elem)
        if i <= new_data.shape[0]:
            new_data = np.delete(new_data, elem, axis=0)
    # fill histogram array
    print(new_data.shape)
    # FIXME index 101 is out of bounds ?????
    for i in range(new_data.shape[0]):
            hist[new_data[i, 2]] += 1

    fig = plt.figure()
    plt.bar(x, hist)
    plt.show(fig)
except (FileNotFoundError):
    print("CSV file not found")
