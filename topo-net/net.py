'''

luke lombardi, april 2016

lightweight f.f. neural network class for embedded systems

alpha  -> beta = input -> hidden
beta -> gamme = hidden -> output

'''


import numpy as np
import random
import math

#constants for randomized initialization
hiddenBiasRange = (-10.0, 10.0)
outputBiasRange = (-10.0, 10.0)

abWeightRange = (-5.0, 5.0)     
bgWeightRange = (-5.0, 5.0)


#activation methods -- if you don't know which one to use, go with sigmoid

activation = [
            "sigmoid",
            "linear",
            "piecewise_linear",
            "step",
            "comp_loglog",
            "bipolar",
            "bipolar_sigmoid",
            "tanh",
            "lecun_tanh",         
            "hard_tanh",
            "absolute",
            "rectifier"
              ]


class network(object):
    def __init__(self, params, randomize=True):
        self.nInput = params['nInput']
        
        self.nHidden = params['nHidden']
        self.nHiddenLayers = len(self.nHidden)
        
        self.nOutput = params['nOutput']
        
        print "Creating network with architecture {", self.nInput, ',', self.nHidden, ',', self.nOutput, '}\n'
        self.shapeInput = (self.nInput*self.nHidden[0])+self.nInput
        self.shapeHidden = []
        
        for i in range(self.nHiddenLayers):
            if i == (self.nHiddenLayers -1):
                self.shapeHidden.append((self.nHidden[i]*self.nOutput)+(self.nHidden[i]*2))
            else:
                self.shapeHidden.append((self.nHidden[i]*self.nHidden[i+1])+(self.nHidden[i]*2))
                            
        self.shapeOutput = (self.nOutput*2)
        
        #construct basic network arrays and set neuron values to None type
        self.construct()
        
        #initialize randomized weights and biases 
        if randomize:
            self.initialize()
            
        if params['activation'] in activation:
            self.activation = params['activation']
            print "Activation method:", self.activation
        
        
    #create vector representation of network
    
    def construct(self):
        self.inputLayer = np.zeros(self.shapeInput, dtype=float)
        self.hiddenLayer = []
        
        for i in range(self.nHiddenLayers):
            self.hiddenLayer.append(np.zeros(self.shapeHidden[i], dtype=float))

        self.outputLayer = np.zeros(self.shapeOutput, dtype=float)
        
        for n in range(1, self.nInput+1):
            self.setInput(n, None)
        
        for i in range(self.nHiddenLayers):
            for n in range(1, self.nHidden[i]+1):
                self.setHidden((i,n), 0.0)
        
        for n in range(1, self.nOutput+1):
            self.setOutput(n, 0.0)        
        
                
    #set up initial randomized parameters for the network
    
    def initialize(self):
        #randomize hidden bias values
        for i in range(self.nHiddenLayers):
            for n in range(1, self.nHidden[i]+1):
                cRand = random.uniform(hiddenBiasRange[0], hiddenBiasRange[1])
                self.setBias((i,n), cRand)
        

        #randomize output bias values
        for n in range(1, self.nOutput+1):
            cRand = random.uniform(outputBiasRange[0], outputBiasRange[1])
            self.setBias((self.nHiddenLayers,n), cRand)
            
            
        #randomize input -> hidden weight values
        for alpha in range(1, self.nInput+1):
            for beta in range(1, self.nHidden[0]+1):
                cRand = random.uniform(abWeightRange[0], abWeightRange[1])
                self.setWeight((0, alpha, beta), cRand)
                
        
        #randomize hidden -> hidden weight values
        for i in range(1, self.nHiddenLayers):
            for beta in range(0, self.nHidden[i-1]):
                for gamma in range(0, self.nHidden[i]):
                    cRand = random.uniform(bgWeightRange[0], bgWeightRange[1])
                    self.setWeight((i, beta, gamma), cRand)
        
        
        #randomize hidden -> output weight values
        for beta in range(0, self.nHidden[self.nHiddenLayers-1]):
            for gamma in range(0, self.nOutput):
                cRand = random.uniform(bgWeightRange[0], bgWeightRange[1])
                self.setWeight((self.nHiddenLayers, beta, gamma), cRand)
        
        
    
    def feed(self, input):
        if len(input) != self.nInput:
            raise Exception('Input vector must be of length: ', self.nInput)

        for n in range(1,self.nInput+1):
            self.setInput(n, input[n-1])
        
        #input to hidden[0] feedforward
        for alpha in range(1, self.nInput + 1):
            for beta in range(1, self.nHidden[0] + 1):
                
                w = self.getWeight((0,alpha,beta))
                i = self.getInput(alpha)
                

                s = self.getHidden((0, beta))
                s += (w*i)

                self.setHidden((0,beta), s)
                
        for beta in range(1, self.nHidden[0] + 1):
            s = self.getHidden((0, beta))
            b = self.getBias((0, beta))
            
            X = s + b
            output = self.activate(X)
                        
            self.setHidden((0, beta), output)
            
        
        for i in range(1, self.nHidden[0] +1):
            for j in range(1, self.nHidden[i-1]):
                pass
        
        print "Hidden Layer 0:"
        print self.hiddenLayer[0] 
        #hidden[0] -> hidden[n]
        
        #for i in range(self.nHiddenLayers):
         #   for j in range(1, self.nHiddenLayers):
        
                
    def activate(self, X):
        if(self.activation == "sigmoid"):
            return 1/1+math.exp(-X)
        
    
    #helper string overload func    
    def __str__(self):
        netString = ""
        netString +=  "Input layer \n"
        netString += str(self.inputLayer)
        netString += "\n"

        
        for i in range(len(self.hiddenLayer)):
            netString += "\nHidden Layer \n"
            netString += str(i)
            netString += str(self.hiddenLayer[i])
            netString += "\n"
        
        netString += "\nOutput Layer \n"
        netString += str(self.outputLayer)
        netString += "\n"

        
        return netString

            
    def __call__(self, params):
       self.__init__(params)
    
    
    
    #setters
    def setInput(self, n, value):
        self.inputLayer[(self.nInput*self.nHidden[0])+(n-1)] = value
        
    def setHidden(self, n, value):
        if n[0] == (self.nHiddenLayers-1):
            self.hiddenLayer[n[0]][(self.nHidden[n[0]]*self.nOutput)+(self.nHidden[0])+(n[1]-1)] = value
        else:
            self.hiddenLayer[n[0]][(self.nHidden[n[0]]*self.nHidden[n[0]+1])+(self.nHidden[0])+(n[1]-1)] = value

        
    def setOutput(self, n, value):
        self.outputLayer[(self.nOutput)+(n-1)] = value
    
    
    def setBias(self, n, value):
        if(n[0] == self.nHiddenLayers -1 ):
            self.hiddenLayer[n[0]][((n[1]-1)+self.nOutput*n[1])] = value
            
        elif(n[0] < (self.nHiddenLayers - 1) ):
            self.hiddenLayer[n[0]][((n[1]-1)+(self.nHidden[n[0]+1]) * n[1])] = value
            
        elif(n[0] == self.nHiddenLayers):
            self.outputLayer[n[1]-1] = value
            
        #n vector is zero indexed 
      
    def setWeight(self, n, value):        
        if(n[0] == 0):  # input -> hidden
            self.inputLayer[((n[1]-1)*self.nHidden[0]) + n[2]-1] = value
            
        elif(n[0] < (self.nHiddenLayers)): #hidden -> hidden
            self.hiddenLayer[n[0]-1][((n[1])*(self.nHidden[n[0]]+1)) + n[2]] = value
           
        elif(n[0] == self.nHiddenLayers):  #hidden -> output
            self.hiddenLayer[n[0]-1][((n[1])*(self.nOutput+1)) + n[2]] = value

    #getters
    
    def getInput(self, n):
        return self.inputLayer[(self.nInput*self.nHidden[0])+(n-1)]
    
        
    def getHidden(self, n):
        if n[0] == (self.nHiddenLayers-1):
            return self.hiddenLayer[n[0]][(self.nHidden[n[0]]*self.nOutput)+(self.nHidden[0])+(n[1]-1)]
        else:
            return self.hiddenLayer[n[0]][(self.nHidden[n[0]]*self.nHidden[n[0]+1])+(self.nHidden[0])+(n[1]-1)] 
    
        
    def getOutput(self, n):
        return self.outputLayer[(self.nHidden[self.nHiddenLayers-1]*self.nOutput)+(n-1)]
    
    
    
    def getBias(self, n):
        if(n[0] == self.nHiddenLayers -1 ):
            return self.hiddenLayer[n[0]][((n[1]-1)+self.nOutput*n[1])]
            
        elif(n[0] < (self.nHiddenLayers - 1) ):
            return self.hiddenLayer[n[0]][((n[1]-1)+(self.nHidden[n[0]+1]) * n[1])] 
            
        elif(n[0] == self.nHiddenLayers):
            return self.outputLayer[n[1]-1]
    
    
    
    def getWeight(self, n):
        
        if(n[0] == 0):  # input -> hidden
            return self.inputLayer[((n[1]-1)*self.nHidden[0]) + n[2]-1]
            
        elif(n[0] < (self.nHiddenLayers)): #hidden -> hidden
            return self.hiddenLayer[n[0]-1][((n[1])*(self.nHidden[n[0]]+1)) + n[2]]
           
        elif(n[0] == self.nHiddenLayers):  #hidden -> output
            return self.hiddenLayer[n[0]-1][((n[1])*(self.nOutput+1)) + n[2]]
        
        
    