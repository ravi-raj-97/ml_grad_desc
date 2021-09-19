# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class GradientDecent:
# Class uses sklearn to generate Stochastic gradient decsent 
# Constructor for the class.
# Data is loaded and scaled using the min_max_scaler from sklearn.
# Input:
# path - the github URL where the dataset is present.
    def __init__(self, path):
        self.path = path
        self.airfoil_data = pd.read_csv(path, sep='\t',header = None)
        self.columns = ['Frequency','angle_of_attack','chord_length','free_stream_velocity','suction_side_displacement_thickness','scaled_sound_pressure_level']
        self.airfoil_data.columns = self.columns
        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(self.airfoil_data.iloc[:,:-1])
        self.scaled_df = pd.DataFrame(data_normalized , columns= self.columns[:-1])
        self.scaled_df['scaled_sound_pressure_level'] = self.airfoil_data.iloc[:,-1:]


# X is dataframe containing all features 
# Y is dataframe containing target variable
# mse is mean squared error
# rmse is root mean squared error
# r2 is R-squared
    def model_building(self):
        X = self.scaled_df[['Frequency','angle_of_attack','chord_length','free_stream_velocity','suction_side_displacement_thickness']]
        Y = self.scaled_df['scaled_sound_pressure_level']
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=5)
        model = SGDRegressor()
        model.fit(X_train, Y_train)
        y_test_predict = model.predict(X_test)
        self.mse = mean_squared_error(Y_test, y_test_predict)
        self.rmse = (np.sqrt(self.mse))
        self.r2 = r2_score(Y_test, y_test_predict)


if __name__ == "__main__":
    Airfoil_model = GradientDecent('https://raw.githubusercontent.com/ravi-raj-97/ml_grad_desc/master/airfoil_self_noise.dat')
    Airfoil_model.model_building()
    print('RMSE is {}'.format(Airfoil_model.rmse))
    print('R2 score is {}'.format(Airfoil_model.r2))
    print('MSE is {}'.format(Airfoil_model.mse))