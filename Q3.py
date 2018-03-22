import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv

def linear_least_square():
    x1=[]
    xi=[]
    yi=[]
    for j in range(1,51):
        x1.append(1)
        xi.append(j)
        temp = j + random.uniform(-1,1)
        yi.append(temp)
    
    X=np.zeros((2,50))
    X[0]=x1
    X[1]=xi
    
    yi_mat= np.matrix(np.array(yi))
    X_mat= np.matrix(np.array(X))
    
    Xp=pinv(X_mat)
    
    wmin=yi_mat*Xp
    
    wmin_list = []
    wmin_list.append(wmin[0,0])
    wmin_list.append(wmin[0,1])
    print ("w0: ",wmin[0,0]," w1: ",wmin[0,1])
    	
    for j in range (0,50):
        plt.plot(xi[j], yi[j],'go')
    	
    y1 = wmin_list[0]
    y2 = wmin_list[0] + (50 *wmin_list[1])
    	
    x_plot = [0,50]
    y_plot = [y1,y2]
    
    plt.plot(x_plot, y_plot, 'b-')
    plt.xlim(0,50)
    plt.ylim(0,52)
    plt.title("xi,yi with linear least squares fit")
    plt.show()
    return


linear_least_square()
