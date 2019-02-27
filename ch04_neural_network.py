
# coding: utf-8

# In[1]:


import sys,os
sys.path.append(os.curdir+"/deep-learning-from-scratch")
from dataset.mnist import load_mnist

import numpy as np
from PIL import Image
import matplotlib.pylab as plt


# In[2]:


(x_train, t_train), (x_text, t_test) = load_mnist(normalize=True, one_hot_label=True)


# In[3]:


t_train[0]


# In[4]:


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# In[5]:


def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# In[6]:


img_show(x_train[0].reshape(28,28))


# In[7]:


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# In[8]:


x_train.shape[0]


# In[9]:


batch_size = 10
batch_mask = np.random.choice(x_train.shape[0], batch_size)
x_batch = x_train[batch_mask]
x_batch.shape[0]


# In[10]:


# 数値微分 (中心差分)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)


# In[11]:


function_1 = lambda x: 0.01 * x**2 + 0.1*x

numerical_diff(function_1, 5)


# In[12]:


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)


# In[13]:


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # same size of x filled by zero

    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


# In[14]:


function_2 = lambda x: np.sum(x**2)


# In[15]:


numerical_gradient(function_2, np.array([3.0, 4.0]))


# In[16]:


# 勾配降下法
def gradient_decent(f, init_x, lr=0.01, step_num=100):    
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x


# In[17]:


init_x = np.array([-3.0, 4.0])
gradient_decent(function_2, init_x, lr=0.5, step_num=1000)


# In[18]:


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # ガウス分布
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = soft_max(z)
        loss = cross_entropy_error(y, t)
        
        return loss


# In[19]:


net = simpleNet()
net.W


# In[20]:


x = np.array([0.6, 0.9])
p = net.predict(x)
np.argmax(p)


# In[21]:


t = np.array([1,0,0]) # 正解ラベル
net.loss(x, t)


# In[22]:


def numerical_gradient2(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad


# In[23]:


f = lambda w: net.loss(x, t)
dW = numerical_gradient2(f, net.W)
dW


# In[37]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[62]:


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b2"] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        
        return loss
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(x, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        
        grads["W1"] = numerical_gradient2(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient2(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient2(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient2(loss_W, self.params["b2"])
        
        return grads


# In[63]:


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params["b1"].shape


# In[64]:


(x_train, t_train), (x_text, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []


# In[65]:


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


# In[66]:


x = np.random.randn(100, 784)
t = np.random.randn(100, 10)

grads = net.numerical_gradient(x, t)
grads


# In[60]:


# hyper parameter
iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print("i = %s" % i)
    print(loss)


train_loss_list


# In[ ]:




