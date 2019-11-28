import numpy as np 
import argparse
from matplotlib import pyplot as plt 
from matplotlib import cm
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
#RBF is the squared exponential kernel but only contains length-scaling(l)
#use constant kernel for signal variance
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, ExpSineSquared

def f(x):
    return np.sin(3*x)

def mean_squared_error(data1, data2):
    se = (data1 - data2)**2
    mean = np.mean(se)
    mse = np.sqrt(mean)
    return mse

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 2 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--')#, label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')

    plt.plot(X, f(X), 'g', lw=1, ls='--', label = 'sin3x')
    plt.legend()
    plt.show()

# Read training data from file
x_train = []
y_train = []
with open('./problem4a_train.csv') as f_csv:
    readCSV = csv.reader(f_csv, delimiter=',')
    for row in readCSV:
        a = float(row[0])
        b = float(row[1])
        x_train.append(a)
        y_train.append(b)

x_train = np.array(x_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)

# Read test input locations
x_test = []
with open('./problem4a_test.csv') as f_test:
    readCSV = csv.reader(f_test)
    for row in readCSV:
        x = float(row[0])
        x_test.append(x)

x_test = np.array(x_test).reshape(-1,1)

# define mean-squared kernel
rbf = ConstantKernel(1.0) * RBF(length_scale = 1.0) + WhiteKernel(1.0,(1e-3,1e+3))
sine_squared_kernel = ExpSineSquared(length_scale = 1.0, periodicity = 1.0)

Parser = argparse.ArgumentParser()
Parser.add_argument('--kernel', default='sine-squared', help='kernel functions that can be used - sine-squared, rbf')
Args = Parser.parse_args()
kernel = Args.kernel

if (kernel == 'rbf'):
    gp = GaussianProcessRegressor(kernel = rbf, alpha=0.01, n_restarts_optimizer = 10)
else:
    gp = GaussianProcessRegressor(kernel = sine_squared_kernel, alpha=0.01, n_restarts_optimizer = 10)

# fit gp using training data
gp.fit(x_train, y_train)

# find mean and covariance at each point for the posterior
mu, cov = gp.predict(x_test, return_cov = True)

# find optimized kernel parameters 
print("Set kernel: ",gp.kernel_)

#Compute mean-squared error
mse = mean_squared_error(mu, f(x_test))
print ("Mean squared error: ", mse)

# plot the gp results
plot_gp(mu, cov, x_test, X_train = x_train, Y_train = y_train)

