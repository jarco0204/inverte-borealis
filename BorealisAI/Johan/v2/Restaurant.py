import datetime
import random
import numpy as np

# GLOBALS
START_DATE = datetime.datetime(2020, 1, 1)
NORMALWORKHOURS = ["Monday", "Tuesday",
                   "Wednesday", "Thursday", "Friday", "Saturday"]
ORDERSPERDAY = 50  # 50 orders average for day
PLATENAMES = {0: "chicken-bacon", 1: "stake-cheese",
              2: "SO-chicken-teriyaki", 3: "meatballs", 4: "tuna"}  # assume that they are bought in same frequency
PLATENINGREDIENTS = {"chicken-bacon": ["chicken-pieces"], "stake-cheese": ["steak-strands", "shredded-cheese"],
                     "SO-chicken-teriyaki": ["chicken-teriyaki"], "meatballs": ["meatballs"], "tuna": ["tuna"]}


VEGETABLENAMES = {0: "avocado", 1: "tomatoes",
                  2: "olives", 3: "green-peppers"}


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
            return (datetime.timedelta(hours=9) + datetime.timedelta(minutes=30))
        else:
            self.normalWorkHours = False  # to determine how many orders to produce
            return (datetime.timedelta(hours=12))

    def getClosingTime(self):
        if self.normalWorkHours:
            return (datetime.timedelta(hours=21) + datetime.timedelta(minutes=30))
        else:
            # print("Special day")
            return (datetime.timedelta(hours=5))

    """
        Needs to return an array of all the orders that the day is going to have
    """

    def determineRateOfOrders(self):
        # First Factor to consider; normal work hours
        rateOfOrdersPerDay = 0
        if(self.normalWorkHours):
            rateOfOrdersPerDay += 0.5
        else:
            rateOfOrdersPerDay += 0.6  # Sundays get more people

        # Second Factor : Weather

        # Third Factor : Random Phenomena
        ranFloat = random.randint(0, 100)/100.0
        rateOfOrdersPerDay += ranFloat

        return(rateOfOrdersPerDay)

    """
        Uses timeframe and rateOfOrders in order to determine all the orders
    """

    def generateOrdersForDay(self):
        self.numOrders = int(np.ceil(ORDERSPERDAY * self.rateOfOrdersPerDay))
        for indOrder in range(1, self.numOrders+1):
            # Determine the plate
            numPlate = int(
                np.floor(random.randint(0, len(PLATENAMES.keys())-1)))
            timeStamp = self.generateOrderTimeStamp(
                indOrder).strftime("%m/%d/%Y, %H:%M:%S")

            # Determine the vegetables
            numVegetables = int(
                np.floor(random.randint(0, len(VEGETABLENAMES.keys())-1)))
            vegetablesForOrder = []
            for veg in range(numVegetables):
                numVegetable = int(
                    np.floor(random.randint(0, len(VEGETABLENAMES.keys())-1)))
                vegetablesForOrder.append(VEGETABLENAMES[numVegetable])

            # Add order info
            if(len(vegetablesForOrder) == 0):
                self.orderObjects.append([PLATENAMES[numPlate], timeStamp])
            else:
                self.orderObjects.append(
                    [PLATENAMES[numPlate], vegetablesForOrder, timeStamp])

            self.generateWeightFluctuations(self.orderObjects[indOrder-1])

    """
        Generate the order time stamp that goes attached to every order
        Makes use of the timeframe, rateOfOrder, and numOrders
    """

    def generateOrderTimeStamp(self, orderNum):
        seconds = self.timeFrame.seconds
        minutes = 0
        if(self.normalWorkHours):
            hours = seconds//3600

        else:
            hours = seconds//3600
            minutes = (seconds % 3600)//60

        convertedToMinutes = (hours*60) + minutes

        orderMinutes = int(
            np.floor(convertedToMinutes*orderNum/self.numOrders))

        return (self.day + self.openRestaurantTime + datetime.timedelta(minutes=orderMinutes))

    """
        Generate the weight fluctuations for each order
    """

    def generateWeightFluctuations(self, orderObject):

        if(len(orderObject) == 2):
            # no vegetables with the order
            weightFlucMainIngredient = self.generateWFPlate(
                orderObject[0], orderObject[1])

        elif(len(orderObject) == 3):
            # order with vegetables
            weightFlucMainIngredient = self.generateWFPlate(
                orderObject[0], orderObject[1])
            weightFlucToppingsAr = []
            for vegetable in orderObject[1]:
                weightFlucToppingsAr.append(
                    self.generateWFToppings(vegetable, orderObject[2]))

    """
        Generate the weight fluctuation for the main plate
    """

    def generateWFPlate(self, plateName, timeStamp):
        ingredientsUsed = PLATENINGREDIENTS[plateName]
        print(ingredientsUsed)

    """
        Generate the weight fluctuations of the toppings (vegetables)
    """

    def generateWFToppings(self, topping, timeStamp):
        print(topping)
