#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import datetime
import random
import pandas as pd
from dateutil.relativedelta import *
from calendar import monthrange


# ### The following function creates a year-long list of days, each item in the list will be a list containing information relevant to that specific day.
# 
# ### The information for each item is in the following format [[Breakfast, Lunch, Dinner], date of the year, weather condition].
# 
# ### Based on this information, we can determine how many customers made an order depending on the day period (B, L, D), and the weather conditions that location was presenting during that specific day.

# In[11]:


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
            yearData[i][j][0] = distributePeriods(yearData[i][j][0])
            yearData[i][j].append(weather)
    return yearData
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

def distributePeriods(n):

    result = []
    result.append(int((0.15*n)/1))
    result.append(int((0.50*n)/1))
    result.append(int((0.35*n)/1))
    return result

### yearsData[month][day][0][period]
### yearsData[0-11][0-30][0-2] //[0-2] if previous == 0

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
    yearData = DataGenerator()

    dfSubway = pd.DataFrame(yearData.generateData())
    #dfSubway


    # In[16]:


    num_rows = len(dfSubway.index)

    print(num_rows)


    # ### The following functions represent the food pan operations for the 5 different protein types available, and the remaining 4 ingredients. This implementation will change in the next days to have more reusable and practical code.

    # In[17]:
    


    dfSubway["TunaFP"] = tunaFPtotal()
    dfSubway["MeatballFP"] = meatballFPtotal()
    dfSubway["ChickenFP"] = chickenFPtotal()
    dfSubway["SteakFP"] = steakFPtotal()
    dfSubway["ChcknTkiFP"] = chickenTFPtotal()
    dfSubway["CheeseFP"] = cheeseFPtotal()
    dfSubway["TomatoFP"] = tomatoFPtotal()
    dfSubway["OlivesFP"] = olivesFPtotal()
    dfSubway["AvocadoFP"] = avocadoFPtotal()

    #dfSubway


    # ### This data set is organized by hour and alphabetically. Contains same information as the previous dataset

    # In[19]:


    #pd.set_option("display.max_rows", None)

    #dfSubway = dfSubway.sort_values(by='Date',ascending=True)
    #dfSubway


    # In[ ]:




# ### The following Sandwich class is responsible for several of the methods utilized to obtain information regarding the sandwich the customer orders, and its details.
# 
# ### The information generated by this Sandwich class is determined based on information obtained from Subway Restaurants, such as popularity of sandwiches provided, probability a client will request a specific sandwich, the average size requested, probability the ingredients we are analyzing will be chosen, the protein amount and ingredient's amount provided per sandwich.
# 
# ### This class contains the following functionalities: generateSubType, generateSize, generateIngredients, generateIngredWeights, generateProteinWeight, and lastly a compilation of these methods which will generate a sandwich, named generateSandwich.
# 

# In[12]:


class Sandwich:
    
    def __init__(self):
        
        self.name = ''
        self.size = ''
        ##############################
        self.tunaWeight = 0
        self.meatballWeight = 0
        self.chickenWeight = 0
        self.chickenTWeight = 0
        self.steakWeight = 0
        ##############################
        self.ingredients = []
        self.ingredWeights = [0,0,0,0]
        
    def generateSubType(self):
        
        #Do a random value with probability, each number is a sub
        
        subsAvailable = ['Tuna', 'Meatball Marinara', 'Chicken & Bacon Ranch Melt',
                         'Steak & Cheese', 'Sweet Onion Chicken Teriyaki']
        
        subChoice = np.random.choice(subsAvailable, p = [0.35, 0.20, 0.18, 0.15, 0.12 ])
        
        self.name = subChoice
        
    def getSubType(self):
        
        return self.name
    
    def generateSize(self):
        
        subSizes = ['6 inch', '12 inch']
        
        size = np.random.choice(subSizes, p = [0.53, 0.47])
        
        self.size = size
    
    def getSize(self):
        
        return self.size
        
    def generateIngredients(self):
        
        # do a random for each ingredient (except for main protein)
        #assign weights
        
                
        set_cheese = ['Shredded cheese', 'None']
        
        cheese = np.random.choice(set_cheese, p = [0.83, 0.17])
                
        self.ingredients.append(cheese)
            
        set_tomato = ['Tomato', 'None']
        
        tomato = np.random.choice(set_tomato, p = [0.61, 0.39])
        
        self.ingredients.append(tomato)
            
        set_olives = ['Olives', 'None']
        
        olives = np.random.choice(set_olives, p = [0.44, 0.56])
        
        self.ingredients.append(olives)
            
        set_avocado = ['Avocado', 'None']
        
        avocado = np.random.choice(set_avocado, p = [0.17, 0.83])
        
        self.ingredients.append(avocado)
            
    
    def getIngredients(self):
        
        return self.ingredients
    
    
    def genCheeseWeight(self):
        
        if(self.ingredients[0] != 'None'):
            
            if(self.size == '6 inch'):
            
                cheeseAmount = random.randint(10,13)
            
                self.ingredWeights[0] = cheeseAmount
                
            elif(self.size == '12 inch'):
                
                cheeseAmount = random.randint(22,26)
            
                self.ingredWeights[0] = cheeseAmount
            
        else:
            
            self.ingredWeights[0] = 0
            
            
    def getCheeseWeight(self):
        
        return self.ingredWeights[0]

    
    def genTomatoWeight(self):
        
        if(self.ingredients[1] != 'None'):
            
            if(self.size == '6 inch'):
                
                tomatoAmount = random.randint(33,37)
            
                self.ingredWeights[1] = tomatoAmount
            
            elif(self.size == '12 inch'):
                
                tomatoAmount = random.randint(68,72)
            
                self.ingredWeights[1] = tomatoAmount
            
        else:
            
            self.ingredWeights[1] = 0
            
            
    def getTomatoWeight(self):
        
        return self.ingredWeights[1]
    
    
            
    def genOlivesWeight(self):
        
        if(self.ingredients[2] != 'None'):
            
            if(self.size == '6 inch'):
                
                olivesAmount = random.randint(4,7)
            
                self.ingredWeights[2] = olivesAmount
            
            elif(self.size == '12 inch'):
            
                olivesAmount = random.randint(10,13)
            
                self.ingredWeights[2] = olivesAmount
            
        else:
            
            self.ingredWeights[2] = 0
            
            
    def getOlivesWeight(self):
        
        return self.ingredWeights[2]
            
            
    def genAvocadoWeight(self):
        
        if(self.ingredients[3] != 'None'):
            
            if(self.size == '6 inch'):
                
                avocadoAmount = random.randint(34,38)
            
                self.ingredWeights[3] = avocadoAmount
            
            elif(self.size == '12 inch'):
            
                avocadoAmount = random.randint(67,73)
            
                self.ingredWeights[3] = avocadoAmount
            
        else:
            
            self.ingredWeights[3] = 0
            
            
    def getAvocadoWeight(self):
        
        return self.ingredWeights[3]
    
    
    def generateIngredWeights(self):
        
        self.genCheeseWeight()
        self.genTomatoWeight()
        self.genOlivesWeight()
        self.genAvocadoWeight()
        
    
    def getIngredWeights(self):
        
        return self.ingredWeights

##################################################################
# This is for the implementation of the different protein types

    def genTunaWeight(self):
        
        if(self.name == 'Tuna' and self.size == '6 inch'):
            
            tunaAmount = random.randint(72,76)
            
            self.tunaWeight = tunaAmount
            
        elif(self.name == 'Tuna' and self.size == '12 inch'):
            
            tunaAmount = random.randint(145,151)
            
            self.tunaWeight = tunaAmount
            
        else:
            
            self.tunaWeight = 0
            
    def getTunaWeight(self):
        
        return self.tunaWeight
            
            
    def genMeatballWeight(self):
        
        if(self.name == 'Meatball Marinara' and self.size == '6 inch'):
            
            meatballAmount = random.randint(135,143)
            
            self.meatballWeight = meatballAmount
            
        elif(self.name == 'Meatball Marinara' and self.size == '12 inch'):
            
            meatballAmount = random.randint(135,143)
            
            self.meatballWeight = meatballAmount
            
        else:
            
            self.meatballWeight = 0
            
    def getMeatballWeight(self):
        
        return self.meatballWeight
            
    def genChickenWeight(self):
        
        if(self.name == 'Chicken & Bacon Ranch Melt' and self.size == '6 inch'):
            
            chickenAmount = random.randint(68,73)
            
            self.chickenWeight = chickenAmount
            
        elif(self.name == 'Chicken & Bacon Ranch Melt' and self.size == '12 inch'):
            
            chickenAmount = random.randint(139,145)
            
            self.chickenWeight = chickenAmount
            
        else:
            
            self.chickenWeight = 0
            
    def getChickenWeight(self):
        
        return self.chickenWeight
            
            
    def genSteakWeight(self):
        
        if(self.name == 'Steak & Cheese' and self.size == '6 inch'):
            
            steakAmount = random.randint(68,74)
            
            self.steakWeight = steakAmount
            
        elif(self.name == 'Steak & Cheese' and self.size == '12 inch'):
            
            steakAmount = random.randint(139,146)
            
            self.steakWeight = steakAmount
            
        else:
            
            self.steakWeight = 0
            
    def getSteakWeight(self):
        
        return self.steakWeight
            
            
    def genChickenTWeight(self):
        
        if(self.name == 'Sweet Onion Chicken Teriyaki' and self.size == '6 inch'):
            
            chickenTAmount = random.randint(83,87)
            
            self.chickenTWeight = chickenTAmount
            
        elif(self.name == 'Sweet Onion Chicken Teriyaki' and self.size == '12 inch'):
            
            chickenTAmount = random.randint(166,173)
            
            self.chickenTWeight = chickenTAmount
            
        else:
            
            self.chickenTWeight = 0
            
    def getChickenTWeight(self):
        
        return self.chickenTWeight
            
    def generateSandwich(self):
        
        self.generateSubType()
        self.generateSize()
        self.genTunaWeight()
        self.genMeatballWeight()
        self.genChickenWeight()
        self.genSteakWeight()
        self.genChickenTWeight()
        self.generateIngredients()
        self.generateIngredWeights()
        
            


class DataGenerator:

    def generateData(self):
    
        dataFullYear = [] #Where data will be stored and returned
        dataChristian = main() #Christian's function to generate the data for each day regarding customer flow and its conditions
    
        for month in range(len(dataChristian)):
            for day in range(len(dataChristian[month])):
                #temp = 0
                for period in range(len(dataChristian[month][day][0])):
                    #if(temp == 0):
                        #hour = random.randint(9,12)
                        #date = datetime.datetime(2020, (month + 1), (day + 1), hour)
                        #temp = 1
                    #elif(temp == 1):
                        #hour = random.randint(12,18)
                        #date = datetime.datetime(2020, (month + 1), (day + 1), hour)
                        #temp = 2
                    #elif(temp == 2):
                        #hour = random.randint(18,22)
                        #date = datetime.datetime(2020, (month + 1), (day + 1), hour)
                        #temp = 0
                    for order in range(dataChristian[month][day][0][period]):
                        
                        if(period == 0):
                            hour = random.randint(9,11)
                            date = datetime.datetime(2020, (month + 1), (day + 1), hour)
                        elif(period == 1):
                            hour = random.randint(12,17)
                            date = datetime.datetime(2020, (month + 1), (day + 1), hour)
                        elif(period == 2):
                            hour = random.randint(18,21)
                            date = datetime.datetime(2020, (month + 1), (day + 1), hour)
        
                        order = Sandwich()
                        order.generateSandwich()
        
                        dataOneOrder = {
                                        "Date": date.strftime('%b %d, %Y, %H'),
                                        "Location": "Avalon Mall",
                                        "Weather": dataChristian[month][day][2],
                                        "SubType": order.getSubType(),
                                        "TunaWeight": order.getTunaWeight(),
                                        "MeatballWeight": order.getMeatballWeight(),
                                        "ChickenWeight": order.getChickenWeight(),
                                        "SteakWeight": order.getSteakWeight(),
                                        "ChcknTkiWeight": order.getChickenTWeight(),
                                        "Cheese": order.getCheeseWeight(),
                                        "Tomato": order.getTomatoWeight(),
                                        "Olives": order.getOlivesWeight(),
                                        "Avocado": order.getAvocadoWeight(),
                                        }
        
                        dataFullYear.append(dataOneOrder)
        
        return dataFullYear



def currTunaFoodPan(number, tunaFPweight):
    # If weight < 500 refill
    #Add the ingred weights first, then do the iloc
    
    tunaFPweight -= dfSubway.iloc[number, 4] 
    
    if(tunaFPweight > 400 ):
        
        return tunaFPweight
    
    else:
        
        tunaFPweight = 3800
        return tunaFPweight
    
    
def tunaFPtotal():
    #Change loop number to the correct number after you have the dataset ready
    tunaFPweight = 3800
    resultTuna = []
    
    for x in range(num_rows):
    
        value = currTunaFoodPan(x, tunaFPweight)
        resultTuna.append(value)
    
        tunaFPweight = value
        
    return resultTuna

def currMeatballFoodPan(number, meatballFPweight):
    # If weight < 500 refill
    #Add the ingred weights first, then do the iloc
    
    meatballFPweight -= dfSubway.iloc[number, 5] 
    
    if(meatballFPweight > 600 ):
        
        return meatballFPweight
    
    else:
        
        meatballFPweight = 4600
        return meatballFPweight
    
    
def meatballFPtotal():
    #Change loop number to the correct number after you have the dataset ready
    meatballFPweight = 4600
    resultMeatball = []
    
    for x in range(num_rows):
    
        value = currMeatballFoodPan(x, meatballFPweight)
        resultMeatball.append(value)
    
        meatballFPweight = value
        
    return resultMeatball

def currChickenFoodPan(number, chickenFPweight):
    # If weight < 500 refill
    #Add the ingred weights first, then do the iloc
    
    chickenFPweight -= dfSubway.iloc[number, 6] 
    
    if(chickenFPweight > 400 ):
        
        return chickenFPweight
    
    else:
        
        chickenFPweight = 3800
        return chickenFPweight
    
    
def chickenFPtotal():
    #Change loop number to the correct number after you have the dataset ready
    chickenFPweight = 3800
    resultChicken = []
    
    for x in range(num_rows):
    
        value = currChickenFoodPan(x, chickenFPweight)
        resultChicken.append(value)
    
        chickenFPweight = value
        
    return resultChicken

def currSteakFoodPan(number, steakFPweight):
    # If weight < 500 refill
    #Add the ingred weights first, then do the iloc
    
    steakFPweight -= dfSubway.iloc[number, 7] 
    
    if(steakFPweight > 400 ):
        
        return steakFPweight
    
    else:
        
        steakFPweight = 3800
        return steakFPweight
    
    
def steakFPtotal():
    #Change loop number to the correct number after you have the dataset ready
    steakFPweight = 3800
    resultSteak = []
    
    for x in range(num_rows):
    
        value = currSteakFoodPan(x, steakFPweight)
        resultSteak.append(value)
    
        steakFPweight = value
        
    return resultSteak

def currChickenTFoodPan(number, chickenTFPweight):
    # If weight < 500 refill
    #Add the ingred weights first, then do the iloc
    
    chickenTFPweight -= dfSubway.iloc[number, 8] 
    
    if(chickenTFPweight > 500 ):
        
        return chickenTFPweight
    
    else:
        
        chickenTFPweight = 4000
        return chickenTFPweight
    
    
def chickenTFPtotal():
    #Change loop number to the correct number after you have the dataset ready
    chickenTFPweight = 4000
    resultChickenT = []
    
    for x in range(num_rows):
    
        value = currChickenTFoodPan(x, chickenTFPweight)
        resultChickenT.append(value)
    
        chickenTFPweight = value
        
    return resultChickenT

def currCheeseFoodPan(number, cheeseFPweight):
    # If weight < 180 refill
    
    cheeseFPweight -= dfSubway.iloc[number, 9] 
    
    if(cheeseFPweight > 160 ):
        
        return cheeseFPweight
    
    else:
        
        cheeseFPweight = 1550
        return cheeseFPweight
    

def cheeseFPtotal():
    
    cheeseFPweight = 1550

    resultCheese = []

    for x in range(num_rows):
    
        value = currCheeseFoodPan(x, cheeseFPweight)
        resultCheese.append(value)
    
        cheeseFPweight = value
        
    return resultCheese

def currTomatoFoodPan(number, tomatoFPweight):
    # If weight < 180 refill
    
    tomatoFPweight -= dfSubway.iloc[number, 10] 
    
    if(tomatoFPweight > 320 ):
        
        return tomatoFPweight
    
    else:
        
        tomatoFPweight = 2530
        return tomatoFPweight
    

def tomatoFPtotal():
    
    tomatoFPweight = 2530

    resultTomato = []

    for x in range(num_rows):
    
        value = currTomatoFoodPan(x, tomatoFPweight)
        resultTomato.append(value)
    
        tomatoFPweight = value
        
    return resultTomato


def currOlivesFoodPan(number, olivesFPweight):
    # If weight < 60 refill
    
    olivesFPweight -= dfSubway.iloc[number, 11] 
    
    if(olivesFPweight > 60 ):
        
        return olivesFPweight
    
    else:
        
        olivesFPweight = 900
        return olivesFPweight
    

def olivesFPtotal():
    
    olivesFPweight = 900

    resultOlives = []

    for x in range(num_rows):
    
        value = currOlivesFoodPan(x, olivesFPweight)
        resultOlives.append(value)
    
        olivesFPweight = value
        
    return resultOlives

def currAvocadoFoodPan(number, avocadoFPweight):
    # If weight < 100 refill
    
    avocadoFPweight -= dfSubway.iloc[number, 12] 
    
    if(avocadoFPweight > 330 ):
        
        return avocadoFPweight
    
    else:
        
        avocadoFPweight = 2550
        return avocadoFPweight


def avocadoFPtotal():
    
    avocadoFPweight = 2550

    resultAvocado = []

    for x in range(num_rows):
    
        value = currAvocadoFoodPan(x, avocadoFPweight)
        resultAvocado.append(value)
    
        avocadoFPweight = value
        
    return resultAvocado




