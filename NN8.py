import numpy.random as random
import numpy as np
import matplotlib.pyplot as pylab
import math

from cvxopt import matrix,solvers

#pylab.title("Input x")
#x=[]
#for i in range(100):
#    x.append([])
#    x[i].append(random.uniform(0,1))
#    x[i].append(random.uniform(0,1))
#    pylab.plot(x[i][0],x[i][1],'gX')
#    
#pylab.show()

x= np.zeros((100,2))

for j in range(100):
    temp = (1-0)*random.sample(2) + 0         
    x[j] = temp
x=np.matrix(np.array(x))

print(x[0,0])

d=[]
d1=[]
d1n=[]
#
pylab.title("Input d")
for j in range(100):
    if (x[j,1]<((math.sin(10*x[j,0]))/5)+0.3) or ((math.pow((x[j,1]-0.8),2)+math.pow((x[j,0]-0.5),2))<math.pow(0.15,2)):
        d1.append(x[j])
        d.append(1.)
        pylab.plot(x[j,0],x[j,1],'rx')
    else:
        d1n.append(x[j])
        d.append(-1.)
        pylab.plot(x[j,0],x[j,1],'bd')
    
        
pylab.show()

P= np.zeros((100,100))
for i in range(100):
    for j in range(100):
        P[i,j]=(d[i]*d[j])*((((x[i,0]*x[j,0])+(x[i,1]*x[j,1]))+1)**2)


P = matrix(P)
q=1*(np.ones(100))
q = matrix(q)
G=-1*(np.identity(100))

G = matrix(G)
h=np.zeros(100)
h = matrix(h)
d=np.matrix(np.array(d))
A = matrix(d)

b = matrix(0.0)
sol=solvers.qp(P, q, G, h, A, b)



alphas=np.array(sol['x'])
print(alphas)
pos=0
for a in alphas:
    if(a>0):
        pos=pos+1

print("Support vectors ",pos)