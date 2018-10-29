#Avoid useless features
#Remove highly correlated features
#Features must be easy to understand

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs =500
#Height is normally distributed
grey_height = 28 + 4*np.random.randn(greyhounds)
lab_height = 24 + 4*np.random.rand(labs)

plt.hist([grey_height,lab_height], stacked =True, color=['r','b'])
plt.show()