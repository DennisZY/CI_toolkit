from scipy.stats import chi2_contingency
from ci_toolbox import chi_square
import numpy as np

if __name__ == '__main__':
    d = np.array([[37, 49, 23], [150, 100, 57]])
    print(chi2_contingency(d))
    print(chi_square(d))