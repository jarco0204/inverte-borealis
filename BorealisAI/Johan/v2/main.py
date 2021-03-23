from Restaurant import Restaurant
import numpy as np
import json

# Globals
ORDEROUTFILE = './orders.json'


def main():
    for dayIndex in range(0, 1):
        # Constructor sets up the time object and variables
        dailyRestaurant = Restaurant(dayIndex)
        dailyRestaurant.generateOrdersForDay()

        writeOrdersToFile(dailyRestaurant.orderObjects)


def writeOrdersToFile(orders):

    try:
        writer = open(ORDEROUTFILE, "w")
        data = {}

        data["order"] = orders
        json.dump(data, writer)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
