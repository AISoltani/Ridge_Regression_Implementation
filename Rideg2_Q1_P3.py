import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading dataset from pandas lib

X_tr = pd.read_csv("housing_X_train.csv", header=None)
y_tr = pd.read_csv("housing_y_train.csv", header=None)
X_tr = np.array(X_tr)
X_tr = X_tr.T
y_tr = np.array(y_tr)
y_tr = np.reshape(y_tr,-1)

X_te = pd.read_csv("housing_X_test.csv", header=None)
y_te = pd.read_csv("housing_y_test.csv", header=None)
X_te = np.array(X_te)
X_te = X_te.T
y_te = np.array(y_te)
y_te = np.reshape(y_te,-1)

n,d = X_tr.shape
nt,dt = X_te.shape

# Ridge Regression Algorithms
max_pass = int(1e7)
tol = 1e-7
step = 1e-4
# Lambda = 10 ##########################################################
w = np.zeros(d)
b = 0
one = np.ones(n)
one_t = np.ones(nt)
lam = 10 
training_error = []
loss_error = []
test_error = []
B = []
W = []
for i in range(max_pass):
    b = 1/(np.dot(one,one)) * (np.dot(one ,(y_tr - np.dot(X_tr,w)))) 
    del_delb = 1/n * (np.dot(one ,(np.dot(X_tr,w) + b*one - y_tr))) 

    del_delb = 1/n * (np.dot(one ,(np.dot(X_tr,w) + b*one - y_tr))) 
    del_delw = 1/n * (np.dot(X_tr.T,(np.dot(X_tr,w) + b*one - y_tr))) + 2*lam*w 
    B.append(b)
    W.append(w)
    w_pre = w
    w = w - step*del_delw
    train_e = 1/(2*n) * np.linalg.norm(np.dot(X_tr,w) + b*one - y_tr)**2
    training_error.append(train_e)
    train_l = 1/(2*n) * np.linalg.norm(np.dot(X_tr,w) + b*one - y_tr)**2 + lam*np.linalg.norm(w)**2
    loss_error.append(train_l)
    test_e  = 1/(2*nt) * np.linalg.norm(np.dot(X_te,w) + b*one_t - y_te)**2
    test_error.append(test_e)
    if np.linalg.norm(w - w_pre)<= tol:
         break
test_l  = 1/(2*nt) * np.linalg.norm(np.dot(X_te,w) + b*one_t - y_te)**2 + lam*np.linalg.norm(w)**2

# Figures
plt.figure()
plt.plot(range(np.array(training_error).shape[0]), np.array(training_error),  linewidth = 4, label = 'Training_error')
plt.plot(range(np.array(loss_error).shape[0]), np.array(loss_error), label = 'Loss_error')
plt.plot(range(np.array(test_error).shape[0]), np.array(test_error), 'g', label = 'Test_error')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title("Step size=1e-4, lambda=10, Tol=1e-7")
plt.show()

# plt.figure()
# plt.plot(range(np.array(B).shape[0]), np.array(B),  linewidth = 2)
# plt.legend()
# plt.xlabel('Number of iterations')
# plt.ylabel('b')
# plt.title("Step size=1e-4, lambda=10")
# plt.show()


del W 
del B 
del training_error 
del loss_error 
del test_error
# Ridge Regression Algorithms
# Lambda = 0 ############################################
w = np.zeros(d)
b = 0
one = np.ones(n)
one_t = np.ones(nt)
lam = 0 
training_error = []
loss_error = []
test_error = []
B = []
W = []
for i in range(max_pass):
    b = 1/(np.dot(one,one)) * (np.dot(one ,(y_tr - np.dot(X_tr,w)))) 
    del_delb = 1/n * (np.dot(one ,(np.dot(X_tr,w) + b*one - y_tr))) 

    del_delb = 1/n * (np.dot(one ,(np.dot(X_tr,w) + b*one - y_tr))) 
    del_delw = 1/n * (np.dot(X_tr.T,(np.dot(X_tr,w) + b*one - y_tr))) + 2*lam*w 
    B.append(b)
    W.append(w)
    w_pre = w
    w = w - step*del_delw
    train_e = 1/(2*n) * np.linalg.norm(np.dot(X_tr,w) + b*one - y_tr)**2
    training_error.append(train_e)
    train_l = 1/(2*n) * np.linalg.norm(np.dot(X_tr,w) + b*one - y_tr)**2 + lam*np.linalg.norm(w)**2
    loss_error.append(train_l)
    test_e  = 1/(2*nt) * np.linalg.norm(np.dot(X_te,w) + b*one_t - y_te)**2
    test_error.append(test_e)
    if np.linalg.norm(w - w_pre)<= tol:
         break
test_l  = 1/(2*nt) * np.linalg.norm(np.dot(X_te,w) + b*one_t - y_te)**2 + lam*np.linalg.norm(w)**2


# Figures
plt.figure()
plt.plot(range(np.array(training_error).shape[0]), np.array(training_error),  linewidth = 4, label = 'Training_error')
plt.plot(range(np.array(loss_error).shape[0]), np.array(loss_error), label = 'Loss_error')
plt.plot(range(np.array(test_error).shape[0]), np.array(test_error), 'g', label = 'Test_error')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title("Step size=1e-4, lambda=0, Tol=1e-7")
plt.show()

# plt.figure()
# plt.plot(range(np.array(B).shape[0]), np.array(B),  linewidth = 2)
# plt.legend()
# plt.xlabel('Number of iterations')
# plt.ylabel('b')
# plt.title("Step size=1e-4, lambda=0")
# plt.show()
