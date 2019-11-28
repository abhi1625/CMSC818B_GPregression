import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import cm
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
#RBF is the squared exponential kernel but only contains length-scaling(l)
#use constant kernel for signal variance
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

def mean_squared_error(data1, data2):
    se = (data1 - data2)**2
    mean = np.mean(se)
    mse = np.sqrt(mean)
    return mse

# Read training data from file
T_list = []
ap_list = []
rh_list = []
v_list = []
pe_list = []
with open('./HW2/problem4b_train.csv') as f_csv:
    readCSV = csv.reader(f_csv, delimiter=',')
    for row in readCSV:
        T = float(row[0])
        ap = float(row[1])
        rh = float(row[2])
        v = float(row[3])
        pe = float(row[4])
        T_list.append(T)
        ap_list.append(ap)
        rh_list.append(rh)
        v_list.append(v)
        pe_list.append(pe)

x_train = np.array([T_list,ap_list,rh_list,v_list]).T
# print(x_train.shape)
# input('a')
y_train = np.array(pe_list).reshape(-1,1)
# print(x_train.shape)
# print(y_train.shape)
# input('a')

# Read test input locations
x_test = []
with open('./problem4b_test.csv') as f_test:
    readCSV = csv.reader(f_test)
    for row in readCSV:
        T_test = float(row[0])
        ap_test = float(row[1])
        rh_test = float(row[2])
        v_test = float(row[3])
        x_test.append([T_test, ap_test, rh_test, v_test])

x_test = np.array(x_test)
# print(x_test.shape, x_test[0])
# input('aaa')

# define mean-squared kernel
rbf = ConstantKernel(1) * RBF(length_scale = 1) + WhiteKernel(1)
gp = GaussianProcessRegressor(kernel = rbf, alpha = 0, n_restarts_optimizer = 10)

# fit gp using training data
gp.fit(x_train[:1000], y_train[:1000])

# find mean and covariance at each point for the posterior
mu, cov = gp.predict(x_test, return_std = True)
print(cov.shape)

with open('./problem4b_output.csv', mode = 'wb') as f_output:
    fileWriter = csv.writer(f_output, delimiter = ',')
    for i in range(mu.shape[0]):
        fileWriter.writerow([mu[i,0], cov[i]])

# find optimized kernel parameters 
print("Set kernel: ",gp.kernel_)

# plot the gp results
pe_sol = []
with open('./problem4b_sol.csv') as f_sol:
    readCSV = csv.reader(f_sol)
    for row in readCSV:
        pe = float(row[0])
        pe_sol.append(pe)

pe_sol = np.array(pe_sol).reshape(-1,1)

mse = mean_squared_error(mu,pe_sol)
print("Mean squared error: ",mse)

plt.plot(pe_sol, mu, 'xr')
plt.plot(pe_sol, pe_sol, '--b')
plt.show()