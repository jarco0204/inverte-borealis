import numpy as np
import datetime
import random
from dateutil.relativedelta import *
from calendar import monthrange

"""
    FIRST TASK: Create 2D array where each cell represents a day.
    @arg year determines which year to create datetime object
"""


def main(year=2020):
    START_DATE = datetime.datetime(year, 1, 1)
    yearData = createOrderPerDaysYearMatrix(START_DATE)  # 2D array

    #print(yearData)
    # Christian's function goes here
    for i in range(len(yearData)):
        for j in range(len(yearData[i])):
            yearData[i][j][0] *= dayOfTheYear(yearData[i][j][1]) * randomMod()
            mod, weather = weatherMod(yearData[i][j][1])
            yearData[i][j][0] *= mod
            yearData[i][j][0] = int(yearData[i][j][0])
            yearData[i][j].append(weather)
"""
    Return array containing each day. Each cell has the format [numOfOrders, timestamp]
    NOTE: Uses relativedelta method from python-dateutil
    @arg the start date
"""
def dayOfTheYear(date):
    dates = {
        '2020-01-01':0.0,
        '2020-02-12':0.8,
        '2020-02-14':0.9,
        '2020-04-02':0.7,
        '2020-04-04':0.5,
        '2020-05-09':0.5,
        '2020-06-20':0.7,
        '2020-07-01':0.8,
        '2020-08-06':1.1,
        '2020-10-11':0.6,
        '2020-10-31':1.3,
        '2020-12-24':1.4,
        '2020-12-25':0.0,
        '2020-12-26':1.1,
        '2020-12-31':1.5
    }
    week = {
        'Monday':0.5,
        'Tuesday':0.4,
        'Friday':1.2,
        'Saturday':1.5,
        'Sunday':1.1,
    }
    option = dates[date.strftime('%Y-%m-%d')] if date.strftime('%Y-%m-%d') in dates else 1
    option *= week[date.strftime('%A')] if date.strftime('%A') in week else 1
    return option

def weatherMod(date):
    winter = {
        'snowStorm':0.1,
        'snowy':0.6,
        'rain':0.9,
        'freezingRain':0.7,
        'clear':1.2
        }
    spring = {
        'rain':0.8,
        'clear':1.1,
        'snowy':0.5
    }
    summer = {
        'rain':0.7,
        'clear':1
    }
    fall = {
        'rain':0.9,
        'clear':1.2,
        'snowy':0.4
    }
    winterM = ['January','February','March','April']
    springM = ['May','June']
    summerM = ['July','August']
    fallM = ['September','October','November','December']
    if date.strftime('%B') in winterM:
        randomWeather = random.random()
        weatherSel = 'snowStorm' if randomWeather < 0.2 else 'rain' if randomWeather >= 0.2 and randomWeather < 0.4 else 'clear' if randomWeather >=0.4 and randomWeather <= 0.6 else 'snowy' if randomWeather > 0.6 and randomWeather <= 0.9 else 'freezingRain'
        mod = winter[weatherSel]
    elif date.strftime('%B') in springM:
        randomWeather = random.random()
        weatherSel = 'snowy' if randomWeather <= 0.2 else 'clear' if randomWeather > 0.2 and randomWeather <= 0.8 else 'rain'
        mod = spring[weatherSel]
    elif date.strftime('%B') in summerM:
        randomWeather = random.random()
        weatherSel = 'clear' if randomWeather <= 0.6 else 'rain'
        mod = summer[weatherSel]
    else:
        randomWeather = random.random()
        weatherSel = 'snowy' if randomWeather <= 0.1 else 'clear' if randomWeather > 0.1 and randomWeather <= 0.6 else 'rain'
        mod = fall[weatherSel]
    return mod, weatherSel

def randomMod():
    return random.uniform(0.8,1.2)

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
