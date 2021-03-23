import datetime

import numpy as np
"""
In an effort to make the artificial data more realistic, we have decided to add a new class called PlateOrder
"""
# Globals
# Specific for Subway at the mall
start_date = datetime.datetime(2020, 1, 1, 9, 30, 0)
end_date = datetime.datetime(2020, 12, 31, 23, 59, 59)
closing_restaurant_time = datetime.time(21, 30, 0)


# ----------------------- Start functions ------------
"""
# Design Decision
    # For V2, only considering our Subway_ideal_customer
    # Sundays have different opening time
    # Time Stamp to normalInit hours
"""


def dayRandomInstance(dayIndex):
    day = openRestaurant(dayIndex)
    for order in orders:

        # Section to determine random # of orders and their type
        # numOrders = randomNumberOrders(day)
    orders = foodOrderInstance(day, ranParams)


"""
    Function to create the a random number of for a given day
"""


def randomNumberOrders(day):
    if(day.strftime("%A") == "Sunday"):
        pass
    else:
        pass


"""
    Function to create the timestamp from which food pan scale begins to register fluctuations
"""


def openRestaurant(dayIndex):

    day = start_date + datetime.timedelta(days=dayIndex)  # default start time
    if(day.strftime("%A") == "Sunday"):
        print("Sunday")
        day = day + datetime.timedelta(hours=2) + \
            datetime.timedelta(minutes=30)
    return day


"""
    Function to create an array of specific Plate_Orders of length x
    Affected by the day of the year
"""


def foodOrderInstance(randPlateOrder):
    # section to build the vegerables
    order = []
    order.append(plateNames[randPlateOrder])
    randNumVegetables = int(
        np.floor(random.randint(0, len(vegetablesNames))))
    for ranVege in range(randNumVegetables):
        order.append(vegetablesNames[int(
            np.floor(random.randint(0, len(vegetablesNames)-1)))])
    return order

# seasons = {"winter": [1, 2, 3], "spring": [4, 5, 6],
#            "summer": [7, 8, 9], "fall": [10, 11, 12]}

# opening_restaurant_time = datetime.time(9, 30, 0)


# restaurant_hours = {"monday-saturday": [opening_restaurant_time, closing_restaurant_time],
#                     "sunday": [datetime.time(12, 0, 0), datetime.time(17, 0, 0)]}
# globals about orders
 # # Section to determine the number of orders in a given day
    # if(day.strftime("%A") == "Sunday"):
    #     numOfOrderForDay = int(
    #         np.floor(random.randint(dailyNumOrdersSpecial[0], dailyNumOrdersSpecial[1])))
    #     # timeRange = 12 * 60  # in minutes
    #     # orderTimeFrequency = float(timeRange)/float(numOfOrderForDay)
    # else:
    #     numOfOrderForDay = int(
    #         np.floor(random.randint(dailyNumOrdersMost[0], dailyNumOrdersMost[1])))
    #     # timeRange = 5 * 60
    #     # orderTimeFrequency = float(timeRange)/float(numOfOrderForDay)
    # # Section determines the timespan of each other
    # # print(orderTimeFrequency)
    # # Section to determine the order's plate and vegetable ingredients
    # finalAr = []
    # orderTime = day
    # for order in range(numOfOrderForDay):
    #     # print(foodOrderInstance(
    #     # int(np.floor(random.randint(numberOfPlates[0], numberOfPlates[1])))))
    #     timeSpan = random.randint(8, 12)
    #     orderTime = orderTime + timedelta(minutes=timeSpan)
    #     print(orderTime)
    # # globals
# dailyNumOrdersMost = [50, 60]  # range to calculate num of orders in most days
# dailyNumOrdersSpecial = [25, 40]  # for days with different opening times
# numberOfPlates = [0, 4]  # just tacking 5 sandwitches
