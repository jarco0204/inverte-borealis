## InVerte-Borealis Final Artificial Dataset

# Version 0.0.1

# Authors:
Christian Tellez Castro
Johan Arcos Mendez
Daniel Grillo Pena

# Summary
This project contains the Unified ramdon data generator.

# Assumptions
• Subway restaurant with 4 subways {Chicken Teriyaky, Steak, Tuna, Meatballs} 
• Operates at the mall.
• M-S work from 8:30am-8:30pm
• Su work from 12pm-5:30pm

## How to Run?
* [Technologies] (#technologies)

To run the experiments simultaneously, run the following commands:
```
$ nnictl create --config config.yml
```
You will see the following:
```
$    INFO:  expand searchSpacePath: ./xgbsearchSpace.json to /home/atellezcastr/project/Repo/inverte-client-api-scripts/inverte-scripts/BorealisAI/Unified/./xgbsearchSpace.json 
$    INFO:  expand codeDir: . to /home/atellezcastr/project/Repo/inverte-client-api-scripts/inverte-scripts/BorealisAI/Unified/. 
$    INFO:  Starting restful server...
$    INFO:  Successfully started Restful server!
$    INFO:  Starting experiment...
$    INFO:  Successfully started experiment!
$    ------------------------------------------------------------------------------------
$    The experiment id is eoQBmUW4
$    The Web UI urls are: http://127.0.0.1:8082   http://10.0.0.30:8082   http://172.17.0.1:8082
$    ------------------------------------------------------------------------------------
$
$    You can use these commands to get more information about the experiment
$    ------------------------------------------------------------------------------------
$            commands                       description
$    1. nnictl experiment show        show the information of experiments
$    2. nnictl trial ls               list all of trial jobs
$    3. nnictl top                    monitor the status of running experiments
$    4. nnictl log stderr             show stderr log content
$    5. nnictl log stdout             show stdout log content
$    6. nnictl stop                   stop an experiment
$    7. nnictl trial kill             kill a trial job by id
$    8. nnictl --help                 get help information about nnictl
$    ------------------------------------------------------------------------------------
$    Command reference document https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html
$    ------------------------------------------------------------------------------------
```

And finally:
```
$ nnictl create --config treeconfig.yml --port 8082
```
You will be able to access both experiments running simultaneously with 127.0.0.1:8080 and 127.0.0.1:8082

To stop them run:
```
$ nnictl stop #experimentId
```

## Technologies
The experiments were done using:
*   Python version: 3.7.2
*   Numpy version: 1.19.2
*   nni version: 2.2
*   Pandas version: 1.2.3
*   Scikit-learn version: 0.24.1
*   Torch version: 1.3.1
*   TorchVision version: 0.4.2
*   XGBoost version: 0.90