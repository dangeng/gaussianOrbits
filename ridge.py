import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

x = np.load('x.npy')
y = np.load('y.npy')

# features
x2 = np.square(x)
y2 = np.square(y)
xy = x*y
ones = np.ones(x.shape[0])
X = np.vstack([x2,xy,y2,x,y,ones]).T

#model = LinearRegression(fit_intercept=False)
#model.fit(X, zeros)

# Sovle [General Form] = 1
# and eventually subtract 1 from the intercept term
proj = np.linalg.inv(np.dot(X.T,X) + np.eye(6))
coeff = np.dot(np.dot(proj,X.T),ones)
print('MSE: {}'.format(np.linalg.norm(np.dot(X, coeff) - ones)))

xv = np.linspace(-9, 9, 400)
yv = np.linspace(-5, 5, 400)
xv, yv = np.meshgrid(xv, yv)

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

axes()
plt.contour(xv, yv, xv*xv*coeff[0] + xv*yv*coeff[1] + yv*yv*coeff[2] + xv*coeff[3] + yv*coeff[4] + coeff[5] - 1, [0], colors='k')
plt.scatter(x,y)
plt.show()
