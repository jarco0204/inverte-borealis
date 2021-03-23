import json
import datetime
import random
import numpy as np
import csv
from enum import Enum
class Container():
    def __init__(self, item, w, p):
        self.ingredient = item
        self.fullWeight = w
        self.specificP = p
        self.currentWeight = w
        self.refills = 0
        self.lefovers = 0
        self.overAndUnder = []

    def addRefill(self):
        self.refills+=1
        self.currentWeight = self.fullWeight
    
    def addNightRefill(self):
        self.refills+=1
        self.lefovers += self.currentWeight
        self.currentWeight = self.fullWeight
    def getRefill(self):
        return self.refills
    
    def getLeftOvers(self):
        return self.lefovers
    def getItem(self):
        return self.ingredient
    def getUsage(self, amount):
        self.currentWeight -= amount
        self.overAndUnder.append(((amount * 1 )/self.specificP)-1)
    def getCurrentW(self):
        return self.currentWeight
    def getProportion(self):
        return self.specificP
class Data():
    def __init__(self, initial):
        self.customersY = initial
        self.customersM = self.setCustomersM()
        self.customersD = self.setCustomersD()
        self.nCustomersM = self.toNumberCM()
        self.nCustomersD = self.toNumberCD()
        
        self.isItDone = False
        for keys in self.nCustomersD:
                if 0 in keys:
                    np.where(keys == 0.0, float(random.randint(1,11)), keys)
                
        self.translator()
        if input("Do you like it?(y/n)") == 'y':
            self.customersP = self.setCustomersP()
            self.nCustomersP = self.toNumberCP()
            self.isItDone = True
        
    
    def setCustomersM(self):
        valuesMonth = np.random.dirichlet(np.ones(12), size=1)
        valuesSorted = list(valuesMonth[0])
        valuesSorted.sort()
        return {
            "1":valuesSorted[0],
            "2":valuesSorted[1],
            "3":valuesSorted[4],
            "4":valuesSorted[8],
            "5":valuesSorted[7],
            "6":valuesSorted[11],
            "7":valuesSorted[9],
            "8":valuesSorted[10],
            "9":valuesSorted[3],
            "10":valuesSorted[5],
            "11":valuesSorted[6],
            "12":valuesSorted[2]}

    def setCustomersD(self):
        months = {
            "1":None,
            "2":None,
            "3":None,
            "4":None,
            "5":None,
            "6":None,
            "7":None,
            "8":None,
            "9":None,
            "10":None,
            "11":None,
            "12":None}
        for month in range(1,13):
            customersThisMonth = self.customersM[str(month)]
            if month == 2:
                selectionPerDay = list(np.random.dirichlet(np.ones(28), size=1)[0])
                selectionPerDay.sort()
                selection = []
                weekends = [2,3,8,9,10,15,16,17,22,23,24]
                deadDays = [1,4,5,11,12,18,19,25,26]
                for day in range(1,29):
                    if day in weekends:
                        selection.append(selectionPerDay[-1])
                        selectionPerDay.pop(-1)
                    elif day in deadDays:
                        selection.append(selectionPerDay[0])
                        selectionPerDay.pop(0)
                    else:
                        selection.append(selectionPerDay[len(selectionPerDay)//2])
                months[str(month)] = selection

            elif month == 4 or month == 6 or month == 9 or month == 11:
                selectionPerDay = list(np.random.dirichlet(np.ones(30), size=1)[0])
                selectionPerDay.sort()
                weekends = ["Friday",'Saturday','Sunday']
                deadDays = ['Monday','Tuesday']
                holidays = ['April 04']
                selection = []
                for day in range(1, 31):
                    currentDate = datetime.date(2019, month, day).strftime('%A')
                    if datetime.date(2019, month, day).strftime('%B %d') in holidays:
                        selection.append(selectionPerDay[-1])
                        selectionPerDay.pop(-1)
                    elif currentDate in weekends:
                        selection.append(selectionPerDay[-1])
                        selectionPerDay.pop(-1)
                    elif currentDate in deadDays:
                        selection.append(selectionPerDay[0])
                        selectionPerDay.pop(0)
                    else:
                        selection.append(selectionPerDay[len(selectionPerDay)//2])
                months[str(month)] = selection
            else:
                selectionPerDay = list(np.random.dirichlet(np.ones(31), size=1)[0])
                selectionPerDay.sort()
                selection = []
                weekends = ["Friday",'Saturday','Sunday']
                deadDays = ['Monday','Tuesday']
                holidays = ['October 11', 'January 01', 'July 01', 'December 25' ]
                for day in range(1, 32):
                    currentDate = datetime.date(2019, month, day).strftime('%A')
                    if datetime.date(2019, month, day).strftime('%B %d') in holidays:
                        selection.append(selectionPerDay[-1])
                        selectionPerDay.pop(-1)
                    elif currentDate in weekends:
                        selection.append(selectionPerDay[-1])
                        selectionPerDay.pop(-1)
                    elif currentDate in deadDays:
                        selection.append(selectionPerDay[0])
                        selectionPerDay.pop(0)
                    else:
                        selection.append(selectionPerDay[len(selectionPerDay)//2])
                months[str(month)] = selection
            
    
        return months

    def setCustomersP(self):
        numbers = []
        for key in self.customersM:
            temp = []
            for i in range(len(self.nCustomersD[int(key)-1])):
                temp2 = []
                values= list(np.random.dirichlet(np.ones(3), size=1)[0])
                values.sort()
                temp2.append(values[0])
                temp2.append(values[2])
                temp2.append(values[1])
                temp.append(temp2)
            numbers.append(temp)
        return numbers

    def toNumberCM(self):
        number = []
        for key in self.customersM:
            number.append((self.customersM[key]*self.customersY)//1)
        return number
    def toNumberCD(self):
        number = []
        for key in self.customersM:
            temp = []
            for i in range(len(self.customersD[key])):
                temp.append((self.customersD[key][i]*self.nCustomersM[int(key)-1])//1)
            number.append(temp)
        return number
    def toNumberCP(self):
        numbers = []
        for i in range(len(self.nCustomersD)):
            temp = []
            for j in range(len(self.nCustomersD[i])):
                temp2 = []
                temp2.append((self.customersP[i][j][0]*self.nCustomersD[i][j])//1)
                temp2.append((self.customersP[i][j][1]*self.nCustomersD[i][j])//1)
                temp2.append((self.customersP[i][j][2]*self.nCustomersD[i][j])//1)
                temp.append(temp2)
            numbers.append(temp)
        return numbers
    def translator(self):
        for key in self.customersM:
            print(key,":", self.nCustomersD[int(key)-1])
    def getYearlyClients(self):
        return self.customersY
    def getMonthlyClients(self):
        return self.nCustomersM
    def getDailyClients(self):
        return self.nCustomersD
    def getClientsInAPeriod(self,month,day):
        return self.nCustomersP[month][day]
    def getClientsInADay(self, month):
        return self.nCustomersD[month]
    def getDoneOrNot(self):
        return self.isItDone
    def getDetails(self):
        arrangedData = {
            "totalCustomers": self.customersY,
            "customersPerMonth": self.nCustomersM,
            "customersPerDay": self.nCustomersD
        }
        with open('dataDetails.json','w') as f:
            f.write(json.dumps(arrangedData, indent=2))
        """with open('dataDetails.csv', 'a+', newline='') as f:
            for data in arrangedData:
                field_names = arrangedData.keys()
                writer =csv.DictWriter(f, fieldnames=field_names)
                writer.writerow(data)"""
class Randomizer():
    def __init__(self, customersData):
        
        self.orders = {1:['bacon', 'chickenStrips'], 2:['steak'], 3:['chickeTeriyaki','onions'], 4:['meatBalls'],5:['tuna']}

        self.protein = [Container('bacon',2000, 15), Container('chickePatty',2000, 71),
            Container('chickenStrips',2000, 71), Container('chickenTeriyaki',2000, 85),
            Container('eggOmeletPatty',2000, 85), Container('eggWhitePatty', 2000, 85),
            Container('ham',2000, 57), 
            Container('meatBalls',2000, 139),Container('pastrami',2000, 57),
            Container('roastBeef',2000, 71),Container('steak',2000, 71), 
            Container('tuna',2000, 74), Container('turkeyBreast',2000, 57),
            Container('veggiePatty',2000, 85)]
        self.cheese = [Container('americanCheese',2000, 11),
            Container('cheddarCheese',2000, 14), Container('monterey',2000, 14),
            Container('mozzarella',2000, 14), Container('parmesan',2000, 1),
            Container('pepperjack',2000, 14), Container('provolone',2000, 14),
            Container('smokedCheddar',2000, 14), Container('swissCheese',2000, 14)]
        self.veggies = [Container('bananaPeppers',2000, 4), Container('cucumbers',2000, 14), 
            Container('greenPeppers',2000, 7), Container('jalapeños',2000, 4),
            Container('pickles',2000, 10), Container('lettuce',2000, 21), 
            Container('blackOlives',2000, 3), Container('onions',2000, 7),
            Container('spinach',2000, 7), Container('tomatoes',2000, 35),
            Container('avocado', 2000, 34)]
        
        self.totalOrders = []
        self.nOrders = 0
        orderN = 0
        for month in range(1, 13):
            for day in range(1, len(customersData.getClientsInADay(month-1))+1):
                temp = 0
                for period in customersData.getClientsInAPeriod(month-1, day-1):
                    for customer in range(int(period)):
                        hour = random.randint(9, 12) if temp == 0 else random.randint(12, 17) if temp ==1 else random.randint(17, 22)
                        date = datetime.datetime(2019,month,day,hour)
                        self.generateOrders(date, orderN)
                        orderN +=1
                    temp +=1

                self.refillContainers()
        self.setUpJson()
        self.setUpCSV()
        with open('refillDetails.json','w') as f:
            refills = []    
            for item in self.protein:
                refills.append({str(item.getItem()):item.getLeftOvers(), 'Refills':item.getRefill()})
            
            for item in self.cheese:
                refills.append({str(item.getItem()):item.getLeftOvers(), 'Refills':item.getRefill()})
                
            for item in self.veggies:
                refills.append({str(item.getItem()):item.getLeftOvers(), 'Refills':item.getRefill()})
            f.write(json.dumps(refills, indent=0))

        
    
    def generateOrders(self, date, number):
        ##Check if containers need refilling
        self.checkContainers()

        ## Generate the random order
        randomOrder = random.randint(1,len(self.orders))
        randomVeggies = [self.veggies[random.randint(0, len(self.veggies)-1)] for i in range(0, random.randint(0,6))]
        randomEProteins = [self.protein[random.randint(0, len(self.protein)-1)] for i in range(0, random.randint(0,3))]
        randomCheese = [self.cheese[random.randint(0, len(self.cheese)-1)] for i in range(0,2)]
        accuracies = []
        ## Update the containers
        for i in randomEProteins:
            amountUsed = random.randint(i.getProportion()-5, i.getProportion()+5)
            i.getUsage(amountUsed)
            accuracies.append(((amountUsed * 1 )/i.getProportion())-1)
        for i in randomCheese:
            amountUsed = random.randint(i.getProportion()-5, i.getProportion()+5)
            i.getUsage(amountUsed)
            accuracies.append(((amountUsed * 1 )/i.getProportion())-1)
        for i in randomVeggies:
            amountUsed = random.randint(i.getProportion()-5, i.getProportion()+5)
            i.getUsage(amountUsed)
            accuracies.append(((amountUsed * 1 )/i.getProportion())-1)

        ##Pass the order

        self.addOrder(randomEProteins, randomCheese, randomVeggies, accuracies, date, number)

    def checkContainers(self):
        ## This function checks if the containers have enough or if not, they will be refilled
        for i in range(len(self.protein)):
            if self.protein[i].getCurrentW() < self.protein[i].getProportion():
                self.protein[i].addRefill()
        for i in range(len(self.cheese)):
            if self.cheese[i].getCurrentW() < self.cheese[i].getProportion():
                self.cheese[i].addRefill()
        for i in range(len(self.veggies)):
            if self.veggies[i].getCurrentW() < self.veggies[i].getProportion():
                self.veggies[i].addRefill()
    
    def refillContainers(self):
        for i in range(len(self.protein)):
            self.protein[i].addNightRefill()
        for i in range(len(self.cheese)):
            self.cheese[i].addNightRefill()
        for i in range(len(self.veggies)):
            self.veggies[i].addNightRefill()
    
    def addOrder(self,proteins, cheese, veggies, accuracies, date, number ):
        for i in proteins:
            self.totalOrders.append({'weightingScaleId':self.protein.index(i), 'currentTime':str(date), 'foodItemName':i.getItem(),'portionAccuracy':accuracies[0],'currentWeight': i.getCurrentW(), 'userId':'cAnderson', 'order':number})
        for i in cheese:
            self.totalOrders.append({'weightingScaleId':self.cheese.index(i)+len(self.protein), 'currentTime':str(date), 'foodItemName':i.getItem(),'portionAccuracy':accuracies[1],'currentWeight': i.getCurrentW(), 'userId':'cAnderson', 'order':number})
        for i in veggies:
            self.totalOrders.append({'weightingScaleId':self.veggies.index(i)+len(self.protein)+len(self.cheese), 'currentTime':str(date), 'foodItemName':i.getItem(),'portionAccuracy':accuracies[2],'currentWeight': i.getCurrentW(), 'userId':'cAnderson', 'order':number})

    def setUpJson(self):
        with open('syntheticData2.json','w') as f:
            f.write(json.dumps(self.totalOrders, indent=0))
    def setUpCSV(self):
        with open('syntheticData.csv', 'a+', newline='') as f:
            for order in self.totalOrders:
                field_names = order.keys()
                writer = csv.DictWriter(f, fieldnames= field_names)
                writer.writerow(order)

def main():
    dataFound = True
    while(dataFound):
        business = Data(random.randint(15500, 17500))
        dataFound = False if business.getDoneOrNot() == True else True

    
    business.getDetails()
    test = Randomizer(business)

if __name__ == '__main__':
    main()
