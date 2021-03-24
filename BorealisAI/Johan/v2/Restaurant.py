import datetime
import random
import numpy as np
from Ingredient import Ingredient

# GLOBALS
START_DATE = datetime.datetime(2020, 1, 1)
NORMALWORKHOURS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
ORDERSPERDAY = 50  # 50 orders average for day
PLATENAMES = {
    0: "chicken-bacon",
    1: "stake-cheese",
    2: "SO-chicken-teriyaki",
    3: "meatballs",
    4: "tuna",
}  # assume that they are bought in same frequency


VEGETABLENAMES = {0: "avocado", 1: "tomatoes", 2: "olives", 3: "green-peppers"}
WEIGHIGSCALESIDS = {
    0: "scaleID_0",
    1: "scaleID_1",
    2: "scaleID_2",
    3: "scaleID_3",
    4: "scaleID_4",
}
PLATENINGREDIENTS = {
    "chicken-bacon": ["chicken-pieces", 100, WEIGHIGSCALESIDS[0]],
    "stake-cheese": ["steak-strands", 120, WEIGHIGSCALESIDS[1]],
    "SO-chicken-teriyaki": ["chicken-teriyaki", 120, WEIGHIGSCALESIDS[2]],
    "meatballs": ["meatballs", 130, WEIGHIGSCALESIDS[3]],
    "tuna": ["tuna", 115, WEIGHIGSCALESIDS[4]],
}


class Restaurant:
    def __init__(self, dayNumberOfYear):
        self.day = None
        self.dayOfWeekName = None
        self.createTimeObject(dayNumberOfYear)  # Sets both variables

        # Set both variables
        self.normalWorkHours = None
        self.openRestaurantTime = self.getOpeningTime()

        # print(self.openRestaurantTime)
        # Set one variable
        self.closeRestaurantTime = self.getClosingTime()
        # print(self.closeRestaurantTime)

        self.timeFrame = self.closeRestaurantTime - self.openRestaurantTime
        # print(self.timeFrame)

        # Set one variable
        self.rateOfOrdersPerDay = self.determineRateOfOrders()

        # This variable is set when generateOrdersForDay() gets called
        self.numOrders = None
        self.orderObjects = []

    def createTimeObject(self, day):
        # Date Object begins at 12 am
        # Setting class variable
        self.day = START_DATE + datetime.timedelta(days=day)
        self.dayOfWeekName = self.day.strftime("%A")  # setting class variable

    def getOpeningTime(self):
        if self.dayOfWeekName in NORMALWORKHOURS:
            self.normalWorkHours = True  # to determine how many orders to produce
            return datetime.timedelta(hours=9) + datetime.timedelta(minutes=30)
        else:
            self.normalWorkHours = False  # to determine how many orders to produce
            return datetime.timedelta(hours=12)

    def getClosingTime(self):
        if self.normalWorkHours:
            return datetime.timedelta(hours=21) + datetime.timedelta(minutes=30)
        else:
            # print("Special day")
            return datetime.timedelta(hours=5)

    """
        Needs to return an array of all the orders that the day is going to have
    """

    def determineRateOfOrders(self):
        # First Factor to consider; normal work hours
        rateOfOrdersPerDay = 0
        if self.normalWorkHours:
            rateOfOrdersPerDay += 0.5
        else:
            rateOfOrdersPerDay += 0.6  # Sundays get more people

        # Second Factor : Weather

        # Third Factor : Random Phenomena
        ranFloat = random.randint(0, 100) / 100.0
        rateOfOrdersPerDay += ranFloat

        return rateOfOrdersPerDay

    """
        Uses timeframe and rateOfOrders in order to determine all the orders
    """

    def generateOrdersForDay(self):
        self.numOrders = int(np.ceil(ORDERSPERDAY * self.rateOfOrdersPerDay))
        for indOrder in range(1, self.numOrders + 1):
            # Determine the plate
            numPlate = int(np.floor(random.randint(0, len(PLATENAMES.keys()) - 1)))
            timeStamp = self.generateOrderTimeStamp(indOrder).strftime(
                "%m/%d/%Y, %H:%M:%S"
            )

            # Determine the vegetables
            numVegetables = int(
                np.floor(random.randint(0, len(VEGETABLENAMES.keys()) - 1))
            )
            vegetablesForOrder = []
            for veg in range(numVegetables):
                numVegetable = int(
                    np.floor(random.randint(0, len(VEGETABLENAMES.keys()) - 1))
                )
                vegetablesForOrder.append(VEGETABLENAMES[numVegetable])

            # Add order info
            if len(vegetablesForOrder) == 0:
                self.orderObjects.append([PLATENAMES[numPlate], timeStamp])
            else:
                self.orderObjects.append(
                    [PLATENAMES[numPlate], vegetablesForOrder, timeStamp]
                )

            self.generateWeightFluctuations(self.orderObjects[indOrder - 1], timeStamp)

            # write the order to ordersDataYear
            _writeOrderToFile(self.orderObjects[indOrder - 1])

    """
        Generate the order time stamp that goes attached to every order
        Makes use of the timeframe, rateOfOrder, and numOrders
    """

    def generateOrderTimeStamp(self, orderNum):
        seconds = self.timeFrame.seconds
        minutes = 0
        if self.normalWorkHours:
            hours = seconds // 3600

        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60

        convertedToMinutes = (hours * 60) + minutes

        orderMinutes = int(np.floor(convertedToMinutes * orderNum / self.numOrders))

        return (
            self.day
            + self.openRestaurantTime
            + datetime.timedelta(minutes=orderMinutes)
        )

    """
        Generate the weight fluctuations for each order
    """

    def generateWeightFluctuations(self, orderObject, time):
        # print(time)
        if len(orderObject) == 2:
            # no vegetables with the order

            weightFlucMainIngredient = self.generateWFPlate(
                orderObject[0], orderObject[1]
            )

        elif len(orderObject) == 3:
            # order with vegetables
            # print(orderObject)
            weightFlucMainIngredient = self.generateWFPlate(
                orderObject[0], orderObject[2]
            )
            weightFlucToppingsAr = []
            for vegetable in orderObject[1]:
                weightFlucToppingsAr.append(
                    self.generateWFToppings(vegetable, orderObject[2])
                )

    """
        Generate the weight fluctuation for the main plate
    """

    def generateWFPlate(self, plateName, timeStamp):

        # weightFoodPan = getCurrentWeightFP()
        mainIngredientData = PLATENINGREDIENTS[plateName]

        # print(mainIngredientData)
        correctPortion = mainIngredientData[1]
        weightScaleID = mainIngredientData[2]  # name, correct portion, scaleID

        # if(len(ingredientsUsed) >= 2):
        #     # TODO: generalize with for loop
        #     secondaryIngredientName = ingredientsUsed[1]
        # myIngredient = Ingredient(mainIngredientData[0], mainIngredientData[1])

        weightReading = Ingredient.fluctuateFoodPan(
            correctPortion, weightScaleID
        )  # object with keys scaleID, fluctuation, currentWeightFoodPan
        weightReading["timestamp"] = timeStamp
        weightReading["ingredientName"] = mainIngredientData[0]
        _writeWeightReadingToFile(weightReading)

    """
        Generate the weight fluctuations of the toppings (vegetables)
    """

    def generateWFToppings(self, topping, timeStamp):
        return


"""
    Utility function to write one order to file with year's worth of data
"""


def _writeOrderToFile(orderObject):
    try:
        # read
        with open("./ordersDataYear.txt", "a") as f:

            f.write(str(orderObject) + "\n")
    except:
        print("error 4")
    finally:
        f.close()


"""
    Utility function to write weightFluctuations generated by order
"""


def _writeWeightReadingToFile(weightFluctuation):
    print(weightFluctuation)
    try:
        # read
        f = open("weightReadingsDataYear.txt", "a")
        f.write(str(weightFluctuation) + "\n")
    except:
        print("error 3")
    finally:
        f.close()
