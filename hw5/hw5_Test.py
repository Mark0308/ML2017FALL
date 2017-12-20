import os
import sys
import numpy as np
import pandas as pd
import sys
from hw5_Model import CFModel, rate, CFModel_report, Deep_model

def predict_rating(model, userid, movieid):
    return rate(model, userid-1, movieid - 1)

def main():
    test_data = pd.read_csv(sys.argv[1], usecols = ['UserID', 'MovieID'])

    model = CFModel(6040, 3952, 200)
    model.load_weights('./model/model_fuck.h5')

    recommendations = pd.read_csv(sys.argv[1], usecols = ['TestDataID'])
    recommendations['Rating'] = test_data.apply(lambda x: predict_rating(model, x['UserID'], x['MovieID']), axis = 1)

    recommendations.to_csv(sys.argv[2], index=False, columns=['TestDataID', 'Rating'])

if __name__ == '__main__':
    main()
