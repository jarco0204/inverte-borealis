import numpy as np
import datetime

"""
    Task 1 (Johan) Create an np.array of 12x31 matrix that contains the days for which data should be generated.

    It establishes a baseline number of orders for a specific year.

    @arg year determines which year to create datetime object
"""


def main(year=2020):
    START_DATE = datetime.datetime(year, 1, 1)

    yearOrders = np.zeros((12, 31), dtype=int)
    for month in range(yearOrders.shape[0]):
        for day in range(yearOrders.shape[1]):
            yearOrders[month][day] = 1
    print(yearOrders)


if __name__ == "__main__":
    main()
