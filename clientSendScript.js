import * as fs from 'fs';
import axios from 'axios';

const path = './cheese.json';

const main = async () => {
    let data = JSON.parse(fs.readFileSync(path, 'utf-8')); //dictionary
    let dataAr = data['cheese']; // array
    // Every object inside this for loop will be sent to the API
    for (let i = 0; i < dataAr.length; i++) {
        // Make a request for a user with a given ID
        let postData = dataAr[i];
        //Research what second key of the object does
        let axiosConfig = {
            headers: {
                'Content-Type': 'application/json;charset=UTF-8',
                'Access-Control-Allow-Origin': '*',
            },
        };

        await axios
            .post(
                'http://localhost:3000/ingredient/real-time',
                postData,
                axiosConfig,
            )
            .then(function () {
                console.log('Gets back ');
            })
            .catch(function (error) {
                // handle error
                console.log(error);
                console.log('fails');
            });
    }
};

main();
