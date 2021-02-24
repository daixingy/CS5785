import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import svm

def dec_func(x):
	y = x-0.5
	return y 

x1 = [3.5,3,2,4,1,2,4,4]
x2 = [1.5,4,2,4,4,1,3,1]
y = [1,1,1,1,-1,-1,-1]
yinde=np.arange(0,5,0.1)

fig = plt.figure()
plt.scatter(x1[:5], x2[:5], color="r")
plt.scatter(x1[5:], x2[5:], color="b")
# plt.plot(yinde, dec_func(yinde))
plt.ylim((0,5))
plt.xlim((0,5))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()




# model = svm.SVC(kernel='linear')
# model.fit(x, y)


# for i in range(7):
# 	plt.scatter(x[i][0],x[i][1],color = 'r')

# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()


# xx = np.linspace(xlim[0], xlim[1], 10)
# yy = np.linspace(ylim[0], ylim[1], 10)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = model.decision_function(xy).reshape(XX.shape)

# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none', edgecolors='k')
# plt.show()



