import numpy.random as random
import numpy as np
import matplotlib.pyplot as pylab
import math

n=100
eta=1
resolution=1000
theta = random.uniform(-1,1)

x= np.zeros((n,2))
for j in range(n):
    temp = (1-0)*random.sample(2) + 0         
    x[j] = temp
x=np.matrix(np.array(x))

def weights(m):
    w=[]
    for i in range(2*m):
        w.append(random.uniform(-1,1))
    return w

def signum(g):
    if(g<0):
        act=-1
    else:
        act=1
    return act

def update_center(x,index_c,c_final):
    f=0
    for value in index_c:
        x[value]=c_final[f]
        f+=1
    return x

def kmeans(c,x,m):
    count=0
    flag=1
    while(flag):
        A={}
        for i in range(m):
            A.setdefault(i,[])


        for i in range(len(x)):
            dist=[]
            for j in range(len(c)):
                dist.append(np.linalg.norm(x[i]-c[j]))
            A[np.argmin(dist)].append(x[i])

        out=0.0
        for i in range(m):
            sum=0
            if(len(A[i])!=0):
                for value in A[i]:
                    sum=sum+value
                avg=sum/(len(A[i]))
                diff=np.linalg.norm(avg-c[i])
                c[i]=avg
                out=diff+out
        if out==0.0:
            flag=0

        count+=1
        print("Count",count)
    return c  

def sun_mountain(m,x):
    
    c1= np.zeros((m,2))
    c1n= np.zeros((m,2))
    x1=[]
    x1n=[]
    des=[]
    index_c1=[]
    index_c1n=[]

    p=0
    q=0
    if m==10:
        pylab.title("Classes plot using RBF network for 20 Centers")
    else:
        pylab.title("Classes plot using RBF network for 4 Centers")
    s=0
    t=0
    for j in range(n):
        if (x[j,1]<((math.sin(10*x[j,0]))/5)+0.3) or ((math.pow((x[j,1]-0.8),2)+math.pow((x[j,0]-0.5),2))<math.pow(0.15,2)):
            
            des.append(1)
            if p<m:
                c1[p]=x[j]
                index_c1.append(j)
                p+=1
            else:
                x1.append(x[j])
                pylab.plot(x[j,0],x[j,1],'gX',label='Class +1'if s==0 else "")
                s=1
        else:
            
            des.append(-1)
            if q<m:
                c1n[q]=x[j]
                index_c1n.append(j)
                q+=1
            else:
                x1n.append(x[j])
                pylab.plot(x[j,0],x[j,1],'rX',label='Class -1'if t==0 else "")
                t=1
    return x,x1,x1n,c1,c1n,index_c1,index_c1n,des

def RBF_PTA(m,w,x,des,c_union,acc):
    epoch=0
    flag=1
    variance = np.var(x)
    while(flag):
        errors=0
        actop=[]

        for i in range(n):
            g=0
            for j in range(2*m):
                rbf= math.exp(-1*((np.linalg.norm(x[i]-c_union[j])**2)/(2*(variance**2))))
                g= (w[j]*rbf)+g
            g=g+theta
            actop.append(signum(g))

        for i in range(n):
            for j in range(2*m):
                rbf= math.exp(-1*((np.linalg.norm(x[i]-c_union[j])**2)/(2*(variance**2))))
                w[j]=w[j]+(eta*rbf*(des[i]-actop[i]))

            if des[i]!=actop[i]:
                errors+=1
        
        epoch+=1
        print("Epoch",epoch,"errors",errors,"accuracy",(n-errors),"%")
        if errors==acc:
            flag=0

    h=0
    for x1 in range(resolution):
        for x2 in range(resolution):
            sum=0
            g=0
            x1t=x1/resolution
            x2t=x2/resolution
            arr=np.array([x1t,x2t])
            for j in range(2*m):
                rbf= math.exp(-1*((np.linalg.norm(arr-c_union[j])**2)/(2*(variance**2))))
                g= (w[j]*rbf)+g
            g=g+theta
            if(g < 0.05 and g > -0.05):
                pylab.plot(x1t,x2t,'b.',markersize=3,label='Decision boundary' if h==0 else "")
                h=1
        print("epoch",x1)
    pylab.legend(loc='upper left')
    pylab.show()

def main(m,acc,x):
    xnew,x1,x1n,c1,c1n,index_c1,index_c1n,des=sun_mountain(m,x)
    x1=np.matrix(np.array(x1))
    x1n=np.matrix(np.array(x1n))
    c1_final=kmeans(c1,x1,m)
    c1n_final=kmeans(c1n,x1n,m)

    for i in range(m):
        pylab.plot(c1_final[i,0],c1_final[i,1],'yo',label='Class +1 centers' if i==0 else "")
        pylab.plot(c1n_final[i,0],c1n_final[i,1],'ko',label='Class -1 centers'if i==0 else "")

    xnew=update_center(xnew,index_c1,c1_final)
    xnew=update_center(xnew,index_c1n,c1n_final)

    c_union=np.concatenate((c1_final,c1n_final))

    w=weights(m)

    RBF_PTA(m,w,xnew,des,c_union,acc)

# passing arguments in main(a,b): a=no.of centers in each class, b=total error % for convergence
main(10,0,x) #PTA converging with maximum 100% accuracy for 20 centers
main(2,12,x) #PTA converging with maximum 88% accuracy for 4 centers