from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

#Gather data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
features = data.drop(['INDUS','AGE'],axis=1)


log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices,columns=['PRICE'])

#Creamos un array con todos los promdios de cada target en formato 1 row 11 columns

PTRATIO_INX = 8
RM_INX = 4
CHAS_INX = 2
CHAS_IDX = 2


property_stats = features.mean().values.reshape(1,11)

#Creamos una Linear Regression y obtenemos el MSE y RMSE
regr = LinearRegression().fit(features,target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target,fitted_vals)
RMSE = np.sqrt(MSE)

#Creamos la función para obtener el precio estimado en log
def get_log_estimate(nr_rooms,
                     students_per_classroom,
                     next_to_river=False,
                     high_confidence=True):
    
    #Configurar la propiedad
    property_stats[0][RM_INX] = nr_rooms
    property_stats[0][PTRATIO_INX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_INX] = 1
    else:
        property_stats[0][CHAS_INX] = 0
    
    #Make Prdition
    log_estimate = regr.predict(property_stats)[0][0]
    
    #Calculate Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
        
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
        
    
    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm,ptratio,chas=False,large_range=True):
    
    """
    Estimat the price of a property in boston
    Keyword Argumnts:
    rm -- number of rooms in proprty
    ptratio -- number of studens per teacher in th clasroom for school in area
    chaz -- True if the property is nxet to the river, False otherwis
    large_range -- True if confidence is 95%, False confidence is 68%
    """
    
    if rm < 1 or ptratio < 1:
        print('That is unrealistic try again')
        return
#Transform 1970s dolars to current dollars
    ZILLOW_MEDIAN_PRICE = 583.3
    SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

    log_est,upper,lower,conf = get_log_estimate(rm,ptratio,next_to_river=chas,high_confidence=large_range)

    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_lo = np.e**lower * 1000 * SCALE_FACTOR

    rounded_est = np.around(dollar_est,-3)
    rounded_hi = np.around(dollar_hi,-3)
    rounded_lo = np.around(dollar_lo,-3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_lo} at the lower nd to USD {rounded_hi} at the high end.')