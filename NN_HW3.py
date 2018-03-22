import numpy as np
from math import log
import matplotlib.pyplot as plt
from numpy.linalg import inv

w = [0.35,0.3]
eta = 0.0001
thres = 0.0001
f_list=[]
w_list=[]

def func(w):
    f=-log(1-w[0]-w[1])-log(w[0])-log(w[1])
    return f

def grad(w):
    g = [((1/(1-w[0]-w[1]))-(1/w[0])),((1/(1-w[0]-w[1]))-(1/w[1]))]
    return g

def grad_desc(w,g,eta):
    wnew=[0,0]
    wnew[0]=w[0]-(eta*g[0])
    wnew[1]=w[1]-(eta*g[1])
    return wnew

def hess(w):
    H = np.matrix([[(1/(1-w[0]-w[1])**2) + (1/(w[0])**2),(1/(1-w[0]-w[1])**2)],[(1/(1-w[0]-w[1])**2),(1/(1-w[0]-w[1])**2) + (1/(w[1])**2)]])
    Hi = inv(np.matrix(np.array(H)))
    return Hi

def newton(w,g,Hi):
    d=np.dot(g,Hi)
    w = w-(eta*d)  
    return w

g=grad(w)
f=func(w)
f_list.append(f)
w_list.append(w)
flag=1
count=0
while flag==1:
    wnew=grad_desc(w,g,eta)
    g=grad(wnew)
    fnew=func(wnew)
    f_list.append(fnew)
    subx = abs(wnew[0]-w[0])
    suby = abs(wnew[1]-w[1])
    w_list.append(wnew)
    w=wnew
    count+=1
    if subx<thres and suby<thres :
        flag=0

for j in range(len(w_list)):
    plt.plot(w_list[j],'bo-')
plt.title("Weights trajectory in Domain D")
plt.show()

plt.plot(f_list,'go-')

plt.show()
plt.title("Energy function using Gradient descent method")
print("Convergence with gradient descent:",count)

w = [0.5,0.35]

g=grad(w)
h=hess(w)
f=func(w)
f_list=[]
f_list.append(f)
flag=1
count=0
while flag==1:
    wu=newton(w,g,h)
    gu=grad(wu)
    hu=hess(wu)
    fu=func(wu)
    subx = abs(wu[0]-w[0])
    suby = abs(wu[1]-w[1])
    f_list.append(fu)
    if subx<thres and suby<thres :
        flag=0
    count+=1
    g=gu
    h=hu
    w=wu
    
plt.show()
plt.title("Weights trajectory in Domain D")
plt.plot(f_list,'go-')

plt.title("Energy function trajectory using Newton's method")
plt.show()
print("Convergence with newton's method:",count)

