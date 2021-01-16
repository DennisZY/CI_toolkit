from scipy.stats import power_divergence
from ci_toolbox import gtest
import numpy as np

if __name__ == '__main__':
    d = np.array([[37, 49, 23], [150, 100, 57]])
    print(power_divergence(d, lambda_="log-likelihood"))
    print(gtest(d))