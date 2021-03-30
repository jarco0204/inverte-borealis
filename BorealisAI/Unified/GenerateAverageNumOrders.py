import numpy as np
import datetime

from dateutil.relativedelta import *
from calendar import monthrange

"""
    FIRST TASK: Create 2D array where each cell represents a day.
    @arg year determines which year to create datetime object
"""


def main(year=2020):
    START_DATE = datetime.datetime(year, 1, 1)
    yearData = createOrderPerDaysYearMatrix(START_DATE)  # 2D array

    print(yearData)
    # Christian's function goes here


"""
    Return array containing each day. Each cell has the format [numOfOrders, timestamp]
    NOTE: Uses relativedelta method from python-dateutil
    @arg the start date
"""


def createOrderPerDaysYearMatrix(start):
    # Set variables
    yearAr = []  # 2D array
    timestamp = start
    monthStart = start
    for month in range(1, 12 + 1):
        AVERAGE_ORDERS = 100
        monthAr = []  # will contain 365 [timeStamp, numOrders]
        num_days = monthrange(2020, month)[1]
        for day in range(num_days):
            monthAr.append([AVERAGE_ORDERS, timestamp])
            timestamp = timestamp + datetime.timedelta(days=1)

        yearAr.append(monthAr)
        monthStart += relativedelta(months=+1)
        timestamp = monthStart
    return yearAr
    # print(datetime.DaysInMonth(2020, month))

    # nextMonth = (beginDay + relativedelta(months=+1)).strftime("%b")
    # curDay = beginDay
    # datetime.DaysInMonth(, month);
    # while curMonth != nextMonth:
    #     MonthAr.append([100, curDay])  # orderPerDay & timestamp
    #     curDay = curDay + datetime.timedelta(days=1)  # add one day
    #     nextMonth = curDay.strftime("%b")
    # yearAr.append(MonthAr)
    # beginDay = curDay

    # print(START_DATE.weekday())
    # Return the day of the week as an integer, where Monday is 0 and Sunday is 6.

    # yearOrders = np.zeros((12, 31), dtype=int)
    # for month in range(yearOrders.shape[0]):
    #     for day in range(1, yearOrders.shape[1] + 1):

    #         # yearOrders[month][day] = 1
    #         START_DATE = START_DATE + datetime.timedelta(days=1)
    # print(START_DATE)

    # print(yearOrders)


if __name__ == "__main__":
    main()
