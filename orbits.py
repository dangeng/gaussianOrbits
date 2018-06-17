import numpy as np
import matplotlib.pyplot as plt

e = .5
p = 2
theta = np.linspace(0,2*np.pi, 200)

# Ellipse with eccentricity e
# Axis "length" p
# Offset by .5 angularly
r = e*p/(1-e*np.cos(theta - .5)) 

# transform to cartesian
x = r * np.cos(theta)
y = r * np.sin(theta)

# Add noise
x += np.random.randn(x.shape[0]) / 20
y += np.random.randn(y.shape[0]) / 20

# plot
plt.scatter(x, y)
plt.show()

# saving
np.save('x.npy', x)
np.save('y.npy', y)
