import numpy as np
from sklearn.datasets import make_classification


#dataset
x_train, y_train = make_classification(n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)


y_train=np.reshape(y_train,(-1,1))

#initial weight and bias values 
b_init = 0
w_init = np.array([ 0.0, 0.0])


def sigmoid(zin):
    zout=1.0/(np.exp(-zin)+1.0)
    return zout



#define the cost function 
def costfunc(x,y,w,b):
    n=x.shape[0]
    summed=0.0
    for i in range(n):
        summed+=y[i]*np.log10(sigmoid(np.dot(w,x[i])+b))+(1.0-y[i])*np.log10(sigmoid(1.0-np.dot(w,x[i]+b)))
    summed=(-1.0)*summed/n
    return summed

#computing the gradient
def comp_grad(x,y,w,b):
    m,n=x.shape
    djw=np.zeros((n))
    djb=0.0
    for i in range(m):
        for j in range(n): 
            djw[j]+=(sigmoid(np.dot(w,x[i])+b)-y[i])*x[i,j]
        djb+=(sigmoid(np.dot(w,x[i])+b)-y[i])
    djb=djb/m
    djw=djw/m

    return djw,djb

#gradient descent 
def graddesc(x,y,w,b,costfunction,gradientfunction,alpha,error,maxiterations):
    jinit=costfunction(x,y,w,b)
    for i in range(maxiterations):
        djw,djb=gradientfunction(x,y,w,b)
        wnew=w-alpha*djw
        bnew=b-alpha*djb
        jnew=costfunction(x,y,wnew,bnew)
        if (abs(jnew-jinit) < error):
            break
        else:
            jinit=jnew
            w=wnew
            b=bnew

    return wnew,bnew,costfunction(x,y,wnew,bnew)


#excecuting the gradient descnet linear regresison for out dataset

from sklearn.linear_model import LogisticRegression


x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lrmod=LogisticRegression()
lrmod.fit(x,y)

wpred=lrmod.coef_
bpred=lrmod.intercept_

a=np.array([0.5,1.5])

inw = np.array([0.0,0.0])
inb = 0.
wfin,bfin,jfin=graddesc(x,y,inw,inb,costfunc,comp_grad,1.0e-4,1.0e-5,100000)

print('scikitlearn prediction',sigmoid(np.dot(wpred,a)+bpred),'my prediction',sigmoid(np.dot(wfin,a)+bfin),'actual value',y_train[0])
