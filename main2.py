import numpy as np
#from scipy.optimize import minimize
#import time
#import sympy
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
#import sklearn as skl
#from sklearn import linear_model
#from sklearn.cross_validation import *
import matplotlib.pyplot as plt
import scipy.stats as st
plt.rcParams['font.family'] = 'fantasy'
plt.rcParams['font.fantasy'] = 'Arial'


df = pd.read_excel('baby-w.xlsx')
data = pd.DataFrame(df)
n = 1174
d = n**(1/3)+1
#print(data["Baby_Weight_kg1"])
#print(data["Mother_Heigh_cm1"])
#x = data[["Baby_Weight_kg1"]].values
#y = data["Mother_Heigh_cm1"].values
#y = y.reshape(-1, 1)
#M = linear_model.LinearRegression()
#lm = M.fit(x,y)
#print(' Intercept: \t', 'Coefficients: \n', M.intercept_[0], M.coef_[0][0], M.coef_[0][1])

X = data["Mother_Heigh_cm1"]**2
y = data["Baby_Weight_kg1"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

#print(model.summary())
#остатки
e = model.resid

#delta с крышкой
delta_q = [ ]
b = 0
for i in range(n):
    delta_q.append(b + e[i])
    b += e[i]

#print(delta_q)

x_current = [i/n for i in range(n)]

# эмпирический мост
sigma_q = (e**2).mean() - e.mean()**2

def z(t):
    return delta_q[int(t*(n-1))]/(0.1*np.sqrt(n))

def z0(t):
    return (0.1/sigma_q)*(z(t) - t*z(1))

Z0 = np.zeros(n)
for i in range(n):
    Z0[i] = z0(i/n)
print(Z0)
plt.title("Эмпирический мост Z0")
plt.plot([i/n for i in range(n)], Z0)
#plt.show()

def L(t):
    s = 0

    for i in range(dn[round(t * (len(dn)-1))]):
        #print(X["const"])
        s+= (X["Mother_Heigh_cm1"][i])
        #print('s',s)
    return s/n

def L0(t):
    return (L(t) - t*L(1))

G = ((X["Mother_Heigh_cm1"])**2).mean() - ((X["Mother_Heigh_cm1"])).mean()**2

def k(s, t):
    return min(s,t) - s*t - L0(s) * L0(t)/G

def k0(s, t):
    return k(s, t) - s*k(1,t) - t*k(s,1) + s*t*k(1,1)


dn = [1]
while (dn[-1] < (n-round(n / (n ** (1 / 3) + 1)))):
    dn.append(dn[-1] + round(n / (n ** (1 / 3) + 1)))
dn.append(n)
print(dn)
Q = np.zeros((len(dn)-1, len(dn)-1))
q = np.zeros(len(dn)-1)

for i in range(len(dn)-1):
    for j in range(len(dn)-1):
       # print("tt",k0(dn[i], dn[j]))
        Q[i][j] = k0(i/(len(dn)), j/(len(dn)))

for i in range(len(dn)-1):
    q[i] = Z0[dn[i]]



print(Q)
print(q)
print(len(dn))
s = np.dot(np.dot(q.T, np.linalg.inv(Q)), q)
print(s)

print(1 - st.chi2.cdf(s, round((n ** (1 / 3) + 1))))


print(L(1))
print(((X["Mother_Heigh_cm1"])).mean())
#alpha = 1 - F(q*Q*q)