#!/usr/bin/env python
# coding: utf-8

# # Tarea 1: Entrenar una red Neuronal
# Nombre: Vicente Illanes

# # Parte 1: Neuronas

# In[222]:


import random
import numpy as np

#Clase Perceptron
class Neuron:
    #Contructor de la clase
    #num_w: numero de pesos (uno por cada input)
    #fun: función de activación
    def __init__(self, num_w, fun):
        ws = []
        for i in range(num_w):
            v = round(random.random(),4)
            ws.append(-1 + (2*v)) #valores entre -1 y 1
        self.w= ws
        b =random.random()
        self.b = -2 + (4*b)
        self.f = fun
        self.delta = 0.0
        
    
    
    def feed(self, x):
        #prediccion de la neurona: f(wx + b) = pred
        #se revisa que entrada sea una lista numerica y coincida con el largo de la lista de pesos
        if(len(self.w)==len(x)):
            try: 
                sum = 0
                for i in range(len(self.w)):
                    sum = sum + self.w[i]*x[i]
                sum = sum + self.b
                self.output= self.f.apply(sum)
                return self.output
            except:
                print("Valores del input deben ser numericos")
        else:
            print("No coinciden largos del input y la cantidad de pesos de la neurona")
    
    def train(self, x, real):
        #training de la neurona
        pred = self.feed(x)
        diff = pred - real
        lr = 0.1
        if(diff!=0):        
            for i in range(len(self.w)):
                self.w[i] = self.w[i] + (lr * x[i] * diff)
            self.b = self.b + (lr * diff)
            
    def getW(self):
        return self.w
    
    def get_w(self, j):
        return self.w[j]
    
    def set_w(self, j, value):
        self.w[j] = self.w[j] + value
    
    def getBias(self):
        return self.b
    
    def setBias(self, value):
        self.b = self.b + value
    
    def transferDerivate(self, output):
        return self.f.derivate(output)
    
    def setDelta(self, value):
        self.delta = self.delta + value
        
    def getDelta(self):
        return self.delta 
        
    def getOutput(self):
        return self.output

            


# # Parte 2: Capas de Neuronas

# In[223]:



#Clase para una capa de la red neuronal
#se crea a partir de un numero de imputs, neuronas y una funcion de activacion 
#(que es la misma para todas las neuronas de la capa)
class NeuronLayer:
    #Constructor
    #x_size: tamaño del input
    #numero de neuronas en la capa
    #fun_act: función de activación
    def __init__(self, x_size, num_neuron, fun_act):
        self.input = x_size
        self.list_neurons = []
        for i in range(num_neuron):
            self.list_neurons.append(Neuron(x_size, fun_act))
        
        self.isOutputLayer = 0 #variable importante para Backpropagation
            
    def feed(self, x):
        if(len(x)== self.input):
            y = []
            for i in range(len(self.list_neurons)):
                y.append(self.list_neurons[i].feed(x))
            return y
        else:
            print("No coinciden largos del input y la cantidad de pesos de la neurona")
            
    def getNeurons(self):
        return self.list_neurons
    
           
           
    
    
    
        

        
            


# In[271]:



class NeuronNetwork:
    def __init__(self, num_layers, list_neuron, x_size, y_size, list_functions):
        
        layers = [x_size] + list_neuron 
        
        self.layers =[]
        for i in range(0, num_layers):
   if(i == num_layers-1):
       self.layers.append(NeuronLayer(list_neuron[-1], y_size, list_functions[-1]))
   else:
       self.layers.append(NeuronLayer(layers[i], layers[i+1], list_functions[i]))
         
    
    def feed(self, x):
        h = x
        for i in range(len(self.layers)):
   h = self.layers[i].feed(h)
        return h
    
    def backpropagation(self, y):
        network = self.layers
        for i in reversed(range(len(network))):
   layer = network[i]
   neurons = layer.getNeurons()
   if i != len(network)-1: 
       #Caso Hidden Layer
       for j in range(len(neurons)):
           error = 0.0
           for neuron in network[i+1].getNeurons():
               error += (neuron.get_w(j) * neuron.getDelta())
           neuron = neurons[j]
           output = neuron.getOutput()
           neuron.setDelta(error * neuron.transferDerivate(output))
   else: 
       #Caso OutputLayer
       for j in range(len(neurons)):
           neuron = neurons[j]
           output = neuron.getOutput()
           error =(y[j] - output)
           neuron.setDelta(error * neuron.transferDerivate(output))        
   
    def update_parameters(self, x, lr):
        network= self.layers
        inputs = x
        for i in range(len(network)):
   neurons = network[i].getNeurons()
   outputs = []
   for neuron in neurons:
       weights = neuron.getW()
       for j in range(len(weights)):
           neuron.set_w(j, lr * neuron.getDelta() * inputs[i])
           neuron.setBias(lr * neuron.getDelta())
       
       output = neuron.getOutput()
       outputs.append(output)
   inputs = outputs
   
    def train_network(self, inputs, expected, lr, n_epoch, safe=True):
        for epoch in range(n_epoch):
   sum_error = 0.0
   for i in range(1):
       outputs = self.feed(inputs[i])
       max = -1
       j=-1
       y_real =[0, 0, 0]
       y_real[expected[i][0]]=1                
       for i in range(len(y_real)):
           if safe:
               sum_error += (y_real[i]-outputs[i])**2
   
       self.backpropagation(y_real)
       self.update_parameters(x, lr)
   print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lr, sum_error/len(inputs)))
           
   



# # Funciones de Activación

# In[272]:


import math
#función escalon
class Step:
    def apply(self, x):
        if (x>=0):
            return 1
        else:
            return 0

#funcion sigmoid
class Sigmoid :
    def apply (self, x):
        return 1 / (1 + math.exp(-x))

    def derivate (self, x):
        return self.apply(x) * (1 - self.apply (x))
    
class Tanh:
    def apply(self, x):
        return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    
    def derivate(self,x):
        return 1 - (self.apply(x)*self.apply(x))


# # Dataset

# ### Carga de datos, normalización y one hot encoding

# In[273]:


import pandas as pd
data = pd.read_csv('iris.data', sep=",")
data.columns = ["sepal length","sepal width", "petal length", "petal width", "class"]
data["class"] = data["class"].map(lambda x: 0 if x =="Iris-setosa" else 1 if x == "Iris-versicolour" else 2 if x == "Iris-virginica" else -1)

data_normalized = data
max = data.max()
min = data.min()
for column in data.columns:
    if(column != "class"):
        data_normalized[column] = data[column].map(lambda x: round((x - min[column])/(max[column]-min[column]),4))
        
X = data_normalized[["sepal length","sepal width", "petal length", "petal width"]]
Y = data_normalized[["class"]]
        


# In[274]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=13, stratify=Y)


# In[275]:


step = Step()
sigmoid = Sigmoid()
tanh = Tanh()
red =NeuronNetwork(4,[10, 5, 5],4,3, [sigmoid , tanh , sigmoid, sigmoid ])    


# In[276]:


batch = X_train.values.tolist()
expected = y_train.values.tolist()
red.train_network(batch, expected, 0.1, 15)


# In[277]:


red.feed(batch[0])


# In[ ]:




