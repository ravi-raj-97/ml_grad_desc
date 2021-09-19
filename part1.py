# importing required libraries
import pandas as pd
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from random import seed
import logging
logging.basicConfig(filename='trial_data.log', filemode='a', level=logging.DEBUG)
seed(4)


class gradient_descent_package:
    def __init__(self, path):
        self.path = path
        self.airfoil_data = pd.read_csv(path, sep='\t', header=None)
        self.all_columns = ['bias_term', 'frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity',
                            'suction_side_displacement_thickness', 'scaled_sound_pressure_level']
        self.airfoil_data.columns = self.all_columns[1:]
        self.num_learner_vars = len(self.all_columns[:-1])
        minimum = self.airfoil_data[self.all_columns[1:]].min()
        maximum = self.airfoil_data[self.all_columns[1:]].max()
        i = 0
        for column in self.airfoil_data.iloc[:, :-1]:
            self.airfoil_data[column] = self.airfoil_data[column].apply(
                lambda x: ((x - minimum[i])) / (maximum[i] - minimum[i]))
            i += 1
        self.airfoil_data['bias_term'] = [1.0] * len(self.airfoil_data)
        self.airfoil_data = self.airfoil_data[self.all_columns]

    def h(x, theta):
        return np.matmul(x, theta)

    def cost_function(x, y, theta):
        return ((gradient_descent_package.h(x, theta) - y).T @ (gradient_descent_package.h(x, theta) - y)) / (
                2 * y.shape[0])

    def gradient_descent(x, y, theta, learning_rate, num_epochs):
        m = x.shape[0]
        j_all = []
        for _ in range(num_epochs):
            h_x = gradient_descent_package.h(x, theta)
            cost_ = (1 / m) * (x.T @ (h_x - y))
            theta = theta - (learning_rate * cost_)
            j_all.append(gradient_descent_package.cost_function(x, y, theta))
        return theta, j_all

    def mse_score(pred_vals, actual_vals):
        mse = np.square(np.subtract(actual_vals, pred_vals)).mean()
        return mse

    def mae_score(pred_vals, actual_vals):
        mae = np.abs(np.subtract(actual_vals, pred_vals)).mean()
        return mae

    def rmse_score(pred_vals, actual_vals):
        rmse = np.sqrt(np.square(np.subtract(actual_vals, pred_vals)).mean())
        return rmse

    def r2_score(pred_vals, actual_vals):
        actual_mean = actual_vals.mean()
        rss = np.sum(np.square(np.subtract(actual_vals, pred_vals)))
        tss = np.sum(np.square(np.subtract(actual_vals, actual_mean)))
        r2 = (1 - (rss / tss))
        return r2

    def parameter_identification(x_train, y_train, theta, lr_vals, epoch_vals, x_test, y_test):
        list_of_results = {'lr': [], 'num_epochs': [], 'theta': [], 'mse': [], 'mae': [], 'rmse': [], 'r2': []}
        for i in range(0, len(lr_vals)):
            for j in range(0, len(epoch_vals)):
                th,jhist = gradient_descent_package.gradient_descent(x_train, y_train, theta, lr_vals[i], epoch_vals[j])
                pred_vals = np.dot(x_test, th)
                mse = gradient_descent_package.mse_score(pred_vals, y_test)
                mae = gradient_descent_package.mae_score(pred_vals, y_test)
                rmse = gradient_descent_package.rmse_score(pred_vals, y_test)
                r2 = gradient_descent_package.r2_score(pred_vals, y_test)
                list_of_results['lr'].append(lr_vals[i])
                list_of_results['num_epochs'].append(epoch_vals[j])
                list_of_results['theta'].append(th)
                list_of_results['mse'].append(mse)
                list_of_results['mae'].append(mae)
                list_of_results['rmse'].append(rmse)
                list_of_results['r2'].append(r2)
        return list_of_results


def main():
    model = gradient_descent_package(
        'https://raw.githubusercontent.com/ravi-raj-97/ml_grad_desc/master/airfoil_self_noise.dat')
    feature_data = model.airfoil_data.iloc[:, :-1]
    target_data = model.airfoil_data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.2)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    theta = np.zeros(x_train.shape[1])
    lr_vals = [0.1,0.05,0.01,0.005,0.001]
    epoch_vals = [1000,5000,10000,50000,100000]
    parameter_values = pd.DataFrame(gradient_descent_package.parameter_identification(x_train, y_train, theta, lr_vals,
                                                                         epoch_vals, x_test, y_test))
    best_r2 = max(parameter_values['r2'])
    print(best_r2)
    sub_data = parameter_values[parameter_values['r2']==best_r2]
    selected_num_epochs = max(sub_data['num_epochs'])
    selected_lr = max(sub_data['lr'])
    print(selected_num_epochs, selected_lr)
    th, jall = gradient_descent_package.gradient_descent(x_train, y_train, theta, selected_lr, selected_num_epochs)
    pred_vals = np.dot(x_test, th)
    print(gradient_descent_package.mse_score(pred_vals, y_test))
    print(gradient_descent_package.mae_score(pred_vals, y_test))
    print(gradient_descent_package.rmse_score(pred_vals, y_test))
    print(gradient_descent_package.r2_score(pred_vals, y_test))
    return


if __name__ == '__main__':
    main()
