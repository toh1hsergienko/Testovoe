from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def linear(a, x, b):
    return  a * x + b

x = np.array([0, 1, 2, 3, 4, 5])  
y = np.array([1.2, 2.7, 4.1, 5.5, 7.8, 9.1])  

mas, mat= curve_fit(linear,x,y)

a,b = mas
x_os = np.linspace(min(x), max(x), 6) 
y_os = linear(x_os, a, b)

mse = np.mean((y_os - y)**2)

print(mse)
plt.scatter(x, y) 
plt.plot(x_os, y_os)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()