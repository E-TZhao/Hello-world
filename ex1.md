

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
a = np.eye(5)
```


```python
a
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])




```python
A = pd.DataFrame(a)
```


```python
A
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



<font color=red size=5>1.*Linear regression with one variable* </font>


```python
f = open("ex1data1.txt","r")
#data = f.readlines()
x = []
y = []
for line in f.readlines():
    x.append(eval(line.split(',')[0]))
    y.append(eval(line.split(',')[1].strip()))
f.close()
m = len(x)
```

Plotting the Data


```python
plt.plot(x, y, 'rx')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.title("Figure 1: Scatter plot of training data")
plt.show()
```


![png](output_8_0.png)


Cost Function


```python
x = np.array(x)
y = np.array(y)
theta = [0 ,0]
iterations = 1500
alpha = 0.01
h = theta[0] + theta[1] * x
J = sum((h - y)**2)/(2*m)
```

Gradient Descent


```python
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

```


![png](output_12_0.png)



```python
predict1 = theta[0] + 3.5 * theta[1]
predict2 = theta[0] +7 *theta[1]
print(predict1, predict2)
```

    0.27983710499196013 4.455454735666521
    


```python
theta
```




    [-3.63029143940436, 1.166362350335582]




```python
from mpl_toolkits.mplot3d import Axes3D
```


```python
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
```


```python
Jlist = []
for i in range(100):
    for j in range(100):
        h = theta0_vals[j] + theta1_vals[i] * x
        Jlist.append(sum((h-y)**2/(2*m)))
J_vals = np.array(Jlist).reshape(100, 100)
```


```python
fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(X, Y, J_vals,)
plt.show()
plt.contour(X, Y, np.log(J_vals))
plt.plot(theta[0], theta[1], 'rx')
plt.show()
```


![png](output_18_0.png)



![png](output_18_1.png)



```python
M1 = np.array([1, 2, 3])
M2 = np.array([[1,2],[2,3],[3,4]])
print(M1)
print(M2)
```

    [1 2 3]
    [[1 2]
     [2 3]
     [3 4]]
    


```python
D1 = pd.DataFrame(M1)
D2 = pd.DataFrame(M2)
print(D1)
print(D2)
```

       0
    0  1
    1  2
    2  3
       0  1
    0  1  2
    1  2  3
    2  3  4
    


```python
m1 = np.mat([1, 2, 3])
m2 = np.mat([[1], [2], [3]])
print(m1)
print(m2)
```

    [[1 2 3]]
    [[1]
     [2]
     [3]]
    


```python
int((m1*m2))+3
```




    17


