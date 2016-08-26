'''

lightweight neural network framework for embedded systems
luke lombardi - april 2016

to use:
    1. define network architecture
        NOTE: if you don't want to initialize weights and biases to random values
        pass in randomize=False when you call the constructor, like n1 = network(params, randomize=False)
        this way you can set specific starting values if you would like 

then -> two options, depending on what your goal is:
    a. train and embed (for simple classifiers)
    b. embed network architecture with recurrent training algorithm (for more advanced time based networks)

'''

#string constants 
BR = "\n-------------------------------------------------------\n"


#neural net arch. definition
inputNeurons = 2
hiddenNeurons = [2, 2, 2, 2, 2]
outputNeurons = 1

#import numpy as np
import random


from net import network

def main():
    params = {
    'nInput':inputNeurons,
    'nHidden':hiddenNeurons,
    'nOutput':outputNeurons,
    'activation':'sigmoid'
              }
    
    n1 = network(params)
    #print n1
    
    #n1.setBias(3, -3.124334)
    
    #print n1
    #cRand = random.uniform(-10.0, 10.0)
    #n1.setWeight((1, 2, 2), cRand)
    #print "\n\n\n"
     
    
    #test xor input vector 
    inputVector = [1.0, 1.0]
    
    try:
        n1.feed(inputVector)
    except Exception as e:
        print BR, "Exception: ", e, BR
        
    #print n1

        
if __name__ == "__main__":main()