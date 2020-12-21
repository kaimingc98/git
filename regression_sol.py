import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
########################################################################
# Data loading
########################################################################
np.random.seed(5)  # seed the random number generator
bodyfat = sio.loadmat('bodyfat_data.mat')

x = bodyfat['X']
y = bodyfat['y']

# split into train and test with seeded randomization
train_idx = np.random.choice(np.arange(x.shape[0]), 150, replace=False)
test_idx = list(set(np.arange(x.shape[0])) - set(train_idx))
x_train = x[train_idx, :]
y_train = y[train_idx]
x_test = x[test_idx, :]
y_test = y[test_idx]


########################################################################
# Regularized Least Square
########################################################################
def rls(x, y, lamda):
    n, d = x.shape
    X = x - np.mean(x, 0)
    theta = np.linalg.inv(X.T.dot(X) + n * lamda * np.identity(d)).dot(X.T.dot(y))
    b = np.mean(y) - np.mean(x, 0).dot(theta)
    w = theta
    return w, b


n = x.shape[0]
lamda = 10

w_rls, b_rls = rls(x_train, y_train, lamda)
y_pred = x_test.dot(w_rls) + b_rls

print('Estimated rls parameters:')
print(w_rls, b_rls, '\n')

print('Mean Squared Error:')
print(np.mean((y_pred-y_test)**2), '\n')

print('Predicted response for x=[100, 100]:')
x_inp = np.array([100, 100])[np.newaxis, :]
y_pred = x_inp.dot(w_rls) + b_rls
print(y_pred)


########################################################################
# Plotting data points and learned hyperplanes
########################################################################
plt3d = plt.figure()
ax = plt3d.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(np.linspace(60, 130, 14), np.linspace(85, 120, 7))
pts = np.array([xx.ravel(), yy.ravel()]).T

# data points
ax.plot(x_test[:, 0], x_test[:, 1], y_test[:, 0], color='red', marker='o', markersize=1, linestyle='None')

# rls hyperplane
rls_pred = pts.dot(w_rls) + b_rls
rls_surf = ax.plot_surface(xx, yy, rls_pred.reshape(xx.shape), alpha=0.5, label='rls')

# fixing bug with plot_surface and legend
rls_surf._facecolors2d = rls_surf._facecolors3d
rls_surf._edgecolors2d = rls_surf._edgecolors3d

# set axis labels and legend
ax.set(xlabel='Abdomen circumference', ylabel='Hip circumference (cm)', zlabel='% body fat')
ax.legend(loc='upper right')

plt.show()
