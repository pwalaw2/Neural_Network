import numpy as np
import numpy.random as random
import matplotlib.pylab as pylab
import math
x=[]
for i in range(300):
    x.append(random.uniform(0,1))

v=[]
for i in range(300):
    v.append(random.uniform(-1/10,1/10))

d=[]
for i in range(300):
    d.append(np.sin(20*x[i])+(3*x[i])+v[i])
   
pylab.title("Curve plots")
pylab.plot(x,d,'go')
    
def weights(N,l,u):
    w=[]
    for i in range(N):
        w.append(random.uniform(l,u))
    return w

w1N = weights(24,-10,9.6)
wb1N = weights(24,-6,7.1)
w2N = weights(24,-5.56,6.84)
wb2N = [-0.1]

n=0.01
epoch=0
epoch_list=[]
MSE_list=[]
thres=1
while(thres>0.012):
    MSE=0
    for j in range(300):
        hid=0
        for i in range(24):
            hid = hid + (w2N[i]*np.tanh(wb1N[i]+(w1N[i]*x[j])))
        y=hid+wb2N[0]
        
        wb2N_upd=wb2N[0]+(2*n*(d[j]-y))
        
        w2N_upd=[]
        w1N_upd=[]
        wb1N_upd=[]
        for i in range(24):
            w2N_upd.append(w2N[i]+(2*n*(d[j]-y)*np.tanh(wb1N[i]+(w1N[i]*x[j]))))
            wb1N_upd.append(wb1N[i]+(2*n*(d[j]-y)*w2N[i]*(1-math.pow(np.tanh(wb1N[i]+(x[j]*w1N[i])),2))))
            w1N_upd.append(w1N[i]+(2*n*(d[j]-y)*w2N[i]*(1-math.pow(np.tanh(wb1N[i]+(x[j]*w1N[i])),2))*x[j]))
        wb2N[0]=wb2N_upd
        for i in range(24):
            w2N[i]=w2N_upd[i]
            wb1N[i]=wb1N_upd[i]
            w1N[i]=w1N_upd[i]
        MSE=MSE+math.pow((d[j]-y),2)  
    epoch=epoch+1
    MSE_min=MSE/300
    epoch_list.append(epoch)
    MSE_list.append(MSE_min)
    print(epoch,MSE_min)
    thres=MSE_min
f=[]

for j in range(300):
    hid=0
    for i in range(24):
        hid = hid + (w2N[i]*np.tanh(wb1N[i]+(w1N[i]*x[j])))
    f.append(hid+wb2N[0])

print("wb2N",wb2N)
print("Minimum w1N: ",min(w1N))
print("Maximum w1N: ",max(w1N))

print("Minimum wb1N: ",min(wb1N)) 
print("Maximum wb1N: ",max(wb1N)) 

print("Minimum w2N: ",min(w2N))
print("Maximum w2N: ",max(w2N))    
    
pylab.plot(x,f,'ro')
pylab.show()
pylab.title("Epoch V/S MSE")
pylab.xlim(0,epoch+1)
pylab.plot(epoch_list,MSE_list,'o-')
pylab.show()


    
    
    