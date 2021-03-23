import * as fs from 'fs'; // find better way
const path = './cheese.json';

let timeObj = new Date(2020, 0, 1, 9, 0); // January 1, 2020, at 9 am

const main = () => {
    let data = prepareData(); // data for ingredient table

    fs.writeFileSync(path, JSON.stringify(data), 'utf-8', function (err) {
        if (err) throw err;
    });
};

//Function to create the data
const prepareData = () => {
    let weightFoodPan = 2000; // In grams
    let cheeseData = {};
    let tempObj = [];
    let possibleWeightsFluc = [
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
    ]; // Range of fluctuations
    for (let i = 0; i < 100; i++) {
        // Section 1: Check to see if food pan is empty
        let refill = false;
        if (weightFoodPan < 90) {
            weightFoodPan = 2000; // refill food pan
            refill = true;
        }
        // Section 2: calculate time
        let timeAnswer = createTime(i); // returns obj with time and condition property
        if (timeAnswer.endDay == true) {
            weightFoodPan = 2000; //Start new day with full food pan
            // Next natural question is what happens when food pans residuals
        }

        // Section 3: weight fluctuation with randomness
        let fluct = possibleWeightsFluc[Math.floor(Math.random() * 20)]; // In range of 0-19
        weightFoodPan -= fluct;

        // Section 3: calculate accuracy of portion
        let accuracy;
        let correctPortionWeight = 100; // This value will be stored by the API
        accuracy = correctPortionWeight - fluct; // Numeric change suggested by Muhammad

        // Create one Ingredient object after scale detects weight change
        tempObj.push({
            weighingScaleID: 'subway_1',
            curTime: timeAnswer.time,
            foodItemName: 'cheese',
            correctPortionWeight: correctPortionWeight,
            portionAccuracy: accuracy,
            CurrentWeightFoodPan: weightFoodPan,
            refill: refill,
            userID: 'johanArcos_5680',
            order: i,
        });
        cheeseData['cheese'] = tempObj; // new weight after employee grabs some
    }
    return cheeseData;
};

//Function to get a new date every 10 second
// Note that this is an estimate
const createTime = () => {
    // Section that computes time fluctuation
    let numI = Math.floor(Math.random() * 15 + 6); // In range of 5 - 12.5 minutes
    let multiplier = Math.floor(numI * 50);
    let milliseconds = multiplier * 1000; // Controls how many seconds it adds.

    // Add the time fluctuation
    timeObj = new Date(timeObj.getTime() + milliseconds);

    // Section to check if fluctuation is after 9 pm; if so, move to next day
    let condition;
    if (
        timeObj >
        new Date(
            timeObj.getFullYear(),
            timeObj.getMonth(),
            timeObj.getDate(),
            21,
            timeObj.getMinutes(),
        )
    ) {
        // Next day
        timeObj = new Date(
            timeObj.getFullYear(),
            timeObj.getMonth(),
            timeObj.getDate() + 1,
            9,
            0,
        );
        timeObj = new Date(timeObj.getTime() + milliseconds); // add miliseconds
        condition = true;
    }

    return { time: new Date(timeObj).toLocaleString(), endDay: condition };
};
main(); //Main Function call
