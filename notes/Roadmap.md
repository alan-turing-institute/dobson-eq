This is a regression task. Our target variable is Ozone amount ($X$) and our covariates are solar intensity ($N$) and path length ($\mu$). There are other covariates, which describe external conditions, e.g. time and location, see [[1 Direct Sun Equation#^c55034]]

There are two cases of Ozone measurements: 
1. When the sun is visible - then we know $X(N, \mu)$
2. When the sun is not visible - then we do not know $X(N, \mu)$ and must regress.

A way forward is to set up a parametric model for case 2 and train it on $X$ from case 1. The training data should be obtained for matching external conditions, e.g. time and location. 

There technical complications: 
1. $N$ is an RV with noise structure described in [[2 R-Dial to N-Values]]
2. Data files contain some topical jargon

