import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets 
from math import exp
import random
import time
# the logistic function
def logistic_func(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    # split into positive and negative to improve stability
    g[t>=0.0] = 1.0 / (1.0 + np.exp(-t[t>=0.0])) 
    g[t<0.0] = np.exp(t[t<0.0]) / (np.exp(t[t<0.0])+1.0)
    return g

# function to compute log-likelihood
def neg_log_like(theta, x, y):
    g = logistic_func(theta,x)
    return -sum(np.log(g[y>0.5])) - sum(np.log(1-g[y<0.5]))

# function to compute the gradient of the negative log-likelihood
def log_grad(theta, x, y):
    g = logistic_func(theta,x)
    return -x.T.dot(y-g)
    
def log_grad_SGD(theta, x, y):
    g = logistic_func(theta,x)
    idx=random.randint(0,x.shape[0]-1)
    return -x[idx]*(y[idx]-g[idx])
# implementation of gradient descent for logistic regression
def grad_desc_Newton(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        Hessian=cal_Hessian(theta,x)
        delta= (np.linalg.solve(Hessian,(log_grad(theta, x, y)))) 
        theta = theta -alpha*delta
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
        print("iter is",iter)
        print("\n")

    return theta, np.array(nll_vec),iter
# implementation of gradient descent for logistic regression
def grad_desc_SGD(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    index = [i for i in range(len(y))]  
    random.shuffle(index)
    x=x[index]
    y=y[index]
    while (nll_delta > tol/10) and (iter < maxiter):
    # for i in range(150):
        idx=iter%x.shape[0]
        if(idx==0):
            index = [i for i in range(len(y))]  
            random.shuffle(index)
            x=x[index]
            y=y[index]
        x_rand=x[idx]
        y_rand=y[idx]
        theta = theta - (alpha * log_grad(theta, x_rand, y_rand)) 
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta =np.abs(nll_vec[-2]-nll_vec[-1])
        iter += 1
    return theta, np.array(nll_vec),iter

def grad_desc(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        theta = theta - (alpha * log_grad(theta, x, y)) 
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec), iter


def cal_Hessian(theta,x):
    Hessian=np.zeros((3,3))
    logist=logistic_func(theta,x)
    for i in range(x.shape[0]):
        x_col=x[i,:]
        x_col=x_col[:,np.newaxis]
        Hessian=Hessian+(x_col.dot(x_col.T))*logist[i]*(1-logist[i])
    return Hessian

def cal_Hessian_1(theta,x):
    Hessian=np.zeros((3,3))
    A=np.eye(x.shape[0])
    logist=logistic_func(theta,x)
    for j in range(x.shape[0]):
        H_theta=logist[j]
        A[j,j]=H_theta*(1-H_theta)+0.0001
    Prime=x.T.dot(A).dot(x)
    Hessian=Prime
    return Hessian
# function to compute output of LR classifier
def lr_predict(theta,x):
    # form Xtilde for prediction
    shape = x.shape
    Xtilde = np.zeros((shape[0],shape[1]+1))
    Xtilde[:,0] = np.ones(shape[0])
    Xtilde[:,1:] = x
    return logistic_func(theta,Xtilde)
time_start=time.time()
## Generate dataset    
np.random.seed(2017) # Set random seed so results are repeatable
x,y = datasets.make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=6.0)

## build classifier
# form Xtilde
shape = x.shape
xtilde = np.zeros((shape[0],shape[1]+1))
xtilde[:,0] = np.ones(shape[0])
xtilde[:,1:] = x

# Initialize theta to zero
theta = np.zeros(shape[1]+1)
# Run gradient descent
alpha=0.01
tol = 1e-3
maxiter = 10000000
theta,cost,iter = grad_desc_SGD(theta,xtilde,y,alpha,tol,maxiter)
time_stop=time.time()
time_cost=time_start-time_stop
print("cost is",cost)
print("iter is",iter)
print("time cost is",time_cost)
## Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_delta = (x[:, 0].max() - x[:, 0].min())*0.05 # add 5% white space to border
y_delta = (x[:, 1].max() - x[:, 1].min())*0.05
x_min, x_max = x[:, 0].min() - x_delta, x[:, 0].max() + x_delta
y_min, y_max = x[:, 1].min() - y_delta, x[:, 1].max() + y_delta
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = lr_predict(theta,np.c_[xx.ravel(), yy.ravel()])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot the training points
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

## Show the plot
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Logistic regression classifier")
plt.show()