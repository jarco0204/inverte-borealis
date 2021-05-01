import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import numpy as np
import nni
import time


def main(args):
    data1, data2, data3 = pd.read_json('../Data/data1.json'), pd.read_json(
        '../Data/data2.json'), pd.read_json('../Data/data3.json')

    frames = [data1, data2, data3]
    data = pd.concat(frames, sort=False)
    outputNorm = preprocessing.normalize([np.array(data['output'])])
    data['output'] = outputNorm[0].tolist()
    # print(data.head)
    trainSet, testSet = train_test_split(data, test_size=0.1)

    regressor = xgb.XGBRegressor(
        n_estimators=args['n_estimators'],
        reg_lambda=args['lambda'],
        gamma=0,
        max_depth=args['depth']
    )
    start = time.time()
    regressor.fit(trainSet.iloc[:, :16], trainSet.iloc[:, 16:])
    run_time = time.time() - start
    nni.report_intermediate_result(run_time)
    nni.report_final_result(mean_squared_error(
        testSet.iloc[:, 16:], regressor.predict(testSet.iloc[:, :16])))


if __name__ == '__main__':
    params = {'n_estimators': 100, 'lambda': 1e-1, 'depth': 15}
    params = nni.get_next_parameter()
    main(params)
