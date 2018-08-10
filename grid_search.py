import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet

x = np.load('x.npy')
y = np.load('y.npy')

# features
x2 = np.square(x)
y2 = np.square(y)
x3 = np.power(x, 3)
y3 = np.power(y, 3)
xxy = np.square(x)*y
xyy = np.square(y)*x
xy = x*y
ones = np.ones(x.shape[0])

X = np.vstack([x2,xy,y2,x,y,ones, x3,y3,xxy,xyy]).T
zeros = np.zeros(x.shape[0])

def fit(alpha, l1_ratio):
    model = ElasticNet(fit_intercept=False, l1_ratio=alpha, alpha=l1_ratio)
    model.fit(X, ones)

    coeff = model.coef_
    mse = np.linalg.norm(np.dot(X, coeff) - ones)
    if np.isnan(mse):
        mse = 10e10
    #print('MSE: {}'.format(mse))
    return coeff, mse

def plot(coeff):
    xv = np.linspace(-3, 3, 400)
    yv = np.linspace(-2, 2, 400)
    xv, yv = np.meshgrid(xv, yv)

    def axes():
        plt.axhline(0, alpha=.1)
        plt.axvline(0, alpha=.1)

    axes()
    plt.contour(xv, yv, xv*xv*coeff[0] + xv*yv*coeff[1] + yv*yv*coeff[2] + xv*coeff[3] + yv*coeff[4] + coeff[5] - 1 + coeff[6]*xv*xv*xv + coeff[7]*yv*yv*yv + coeff[8]*xv*xv*yv + coeff[9]*xv*yv*yv , [0], colors='k')
    plt.scatter(x,y)
    plt.show()

mses = np.zeros((100,100))
alphas = np.linspace(.1,2,100)
l1_ratios = np.linspace(.1,1,100)
for i in range(100):
    print(i)
    for j in range(100):
        coeff, mse = fit(alphas[i], l1_ratios[j])
        #plot(coeff)
        mses[i,j] = mse

best_idx = np.unravel_index(mses.argmin(), mses.shape)
coeff, mse = fit(alphas[best_idx[0]], l1_ratios[best_idx[1]])
plot(coeff)
