import random
import json


class Ingredient:
    def __init__(self, name, correctPortion):
        self.name = name
        self.correctPortion = correctPortion
        self.weightFoodPan = 2000  # 2000 Grams

    """
        Receives a -20;+20 weight fluctuation number
    """

    @staticmethod
    def fluctuateFoodPan(correctPortion, weightScaleID):
        fluctuationFactor = random.randrange(-10 + correctPortion, 10 + correctPortion)
        # weightChange = self.correctPortion + fluctuationFactor
        # self.weightFoodPan = self.weightFoodPan - weightChange
        # print(fluctuationFactor)
        # print(weightScaleID)
        # private function to look for current weight of food pan
        curWeightFP = _getCurrentWeightFoodPan(weightScaleID)
        curWeightFP = curWeightFP - fluctuationFactor
        dataDictReturn = {
            "scaleID": weightScaleID,
            "fluctuation": fluctuationFactor,
            "currentWeightFoodPan": curWeightFP,
        }
        if curWeightFP < correctPortion + 10:
            print("refill")
            dataDictReturn["currentWeightFoodPan"] = curWeightFP
            curWeightFP = 2000
        # private function to update the value
        _updateCurrentWeightFoodPan(weightScaleID, curWeightFP)
        return dataDictReturn


def _getCurrentWeightFoodPan(scaleID):
    """
    External function to read from file and return current weight of food pan
    """
    try:
        with open("scalesWeightDataDaily.json") as f:
            data = json.load(f)
            return data[scaleID]
    except:
        print("error")
    finally:
        f.close()


def _updateCurrentWeightFoodPan(scaleID, weight):
    try:
        # read
        jsonData = None
        with open("scalesWeightDataDaily.json") as f:
            data = json.load(f)
            # print(data)
            data[scaleID] = weight
            jsonData = data
        with open("scalesWeightDataDaily.json", "w") as json_file:
            json.dump(jsonData, json_file)

    except:
        print("error1")
    finally:
        f.close()
        json_file.close()
