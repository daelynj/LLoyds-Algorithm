import numpy as np
import csv


def process_data(src):
    data = None
    with open(src, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)

    data = [[float(y) for y in x] for x in data]
    return [np.array(x, dtype='float') for x in data]
