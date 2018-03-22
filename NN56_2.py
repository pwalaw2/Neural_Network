import numpy.random as random
import numpy as np
import matplotlib.pyplot as pylab
import math

from cvxopt import matrix,solvers

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
pylab.title("Input e")
for j in range(100):
    if (x[j,1]<((math.sin(10*x[j,0]))/5)+0.3) or ((math.pow((x[j,1]-0.8),2)+math.pow((x[j,0]-0.5),2))<math.pow(0.15,2)):
        d1.append(x[j])
        d.append(1.)
        pylab.plot(x[j,0],x[j,1],'rx')
    else:
        d1n.append(x[j])
        d.append(-1.)
        pylab.plot(x[j,0],x[j,1],'bd')
    
        


P= np.zeros((100,100))
for i in range(100):
    for j in range(100):
        P[i,j]=(d[i]*d[j])*((((x[i,0]*x[j,0])+(x[i,1]*x[j,1]))+1)**5)


P = matrix(P)
q=-1*(np.ones(100))
q = matrix(q)
G=-1*(np.identity(100))

G = matrix(G)
h=np.zeros(100)
h = matrix(h)
A = matrix(np.matrix(np.array(d)))

b = matrix(0.0)
sol=solvers.qp(P, q, G, h, A, b)



alphas=np.ravel(sol['x'])

# print(alphas)

sv=0

xi1_sup=[]
xi2_sup=[]
d_sup=[]
for i in range(100):
    if(alphas[i]>90):
        sv=sv+1
        xi1_sup.append(x[i,0])
        xi2_sup.append(x[i,1])
        pylab.plot(x[i,0],x[i,1],'ko')
        d_sup.append(d[i])
# print(x_sup)
print("support vectors",sv)
theta=[]
# pylab.show()

for j in range(sv):
    sum=0
    for i in range(100):
        sum=sum+ (alphas[i]*d[i]*((((x[i,0]*xi1_sup[j])+(x[i,1]*xi2_sup[j]))+1)**5))
    theta.append(d_sup[j]-sum)

# print(theta)

for x1 in range(1000):
	for x2 in range(1000):
		sum=0
		g=0
		x1t=x1/1000
		x2t=x2/1000
		for i in range(100):
			sum=sum+ (alphas[i]*d[i]*((((x[i,0]*x1t)+(x[i,1]*x2t))+1)**2))
		g=sum+theta[0]
		if(g < 0.1 and g > -0.1):
			pylab.plot(x1t,x2t,'k.')

		if(g < 1.1 and g > 0.9):
			pylab.plot(x1t,x2t,'c.')

		if(g < -0.9 and g > -1.1):
			pylab.plot(x1t,x2t,'g.')
	print("epoch",x1)

pylab.show()
