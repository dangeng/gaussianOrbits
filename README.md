# gaussianOrbits

Linear regression methods to determine the orbits of heavenly bodies.

Kind of like what Gauss did, but wayyy simplified.

-`gen_data.py` creates noisy orbit `(x,y)` samples and saves them to `x.npy` and `y.npy`

-`ridge.py` ridge regression on quadratic features of the points. Comes up with a good approxination to the orbit in cartesian coordinates

-`lasso.py` runs "Elastic Net" regression on the orbit data on cubic features of the points. The cubic features should zero out and only the quadratic and lower degree monomials should remain. [ADD GRID SEARCH HERE]
