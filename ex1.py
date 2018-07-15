
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


a = np.eye(5)


# In[3]:


a


# In[4]:


A = pd.DataFrame(a)


# In[5]:


A


# <font color=red size=5>1.*Linear regression with one variable* </font>

# In[2]:


f = open("ex1data1.txt","r")
#data = f.readlines()
x = []
y = []
for line in f.readlines():
    x.append(eval(line.split(',')[0]))
    y.append(eval(line.split(',')[1].strip()))
f.close()
m = len(x)


# Plotting the Data

# In[40]:


plt.plot(x, y, 'rx')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.title("Figure 1: Scatter plot of training data")
plt.show()


# Cost Function

# In[4]:


x = np.array(x)
y = np.array(y)
theta = [0 ,0]
iterations = 1500
alpha = 0.01
h = theta[0] + theta[1] * x
J = sum((h - y)**2)/(2*m)


# Gradient Descent

# In[5]:


for i in range(iterations):
    p = theta[0] - alpha/m * sum(h - y)
    q = theta[1] - alpha/m * sum((h - y) * x)
    theta[0] = p
    theta[1] = q
    h = theta[0] + theta[1] * x

plt.plot(x, y, 'rx')
plt.plot(x, h, 'b-')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.title("Plot of training data and linear regression")
plt.legend(labels=["Training data", "Linear regression"])
plt.show()


# In[65]:


predict1 = theta[0] + 3.5 * theta[1]
predict2 = theta[0] +7 *theta[1]
print(predict1, predict2)


# In[6]:


theta


# In[90]:


from mpl_toolkits.mplot3d import Axes3D


# In[127]:


theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)


# In[148]:


Jlist = []
for i in range(100):
    for j in range(100):
        h = theta0_vals[j] + theta1_vals[i] * x
        Jlist.append(sum((h-y)**2/(2*m)))
J_vals = np.array(Jlist).reshape(100, 100)


# In[170]:


fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(X, Y, J_vals,)
plt.show()
plt.contour(X, Y, np.log(J_vals))
plt.plot(theta[0], theta[1], 'rx')
plt.show()


# In[19]:


M1 = np.array([1, 2, 3])
M2 = np.array([[1,2],[2,3],[3,4]])
print(M1)
print(M2)


# In[22]:


D1 = pd.DataFrame(M1)
D2 = pd.DataFrame(M2)
print(D1)
print(D2)


# In[23]:


m1 = np.mat([1, 2, 3])
m2 = np.mat([[1], [2], [3]])
print(m1)
print(m2)


# In[31]:


int((m1*m2))+3

