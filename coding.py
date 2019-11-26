import numpy as np
import math

from channels import Channel, addComplexGaussianNoise

class AlamoutiCoding:
    def __init__(self, nr, channel, generator):
        self.Nr = nr
        self.channel = channel
        self.transmitted = None
        self.toTransmit = None
        self.generator = generator
    
    def code(self, x):
        
        self.originalMessage = x
        x = x.reshape((-1,2))
        
        s = x.shape
        y = np.zeros(shape=(s[0]*2,s[1]), dtype=complex)
        
        for i in range(s[0]):
            y[2*i,:] = x[i,:]
            y[2*i + 1,0] = -np.conj(x[i,1])
            y[2*i + 1,1] = np.conj(x[i,0])
        
        self.toTransmit = y
        return y
    
    def transmit(self,x = None, var = 1, power = None):
        if(x is None):
            if(self.toTransmit is None):
                raise ValueError('Nothing to transmit, encode should be called prior to this method')
            else:
                x = self.toTransmit
        
        if(power is None):
            power = 1.0/self.Nr
        x = x*power

        s = x.shape
        y = np.zeros(shape=(s[0], self.Nr), dtype=complex)
        H = self.channel.getChannel()
        
        for i in range(s[0]):
            yy = np.matmul(H,x[i,:])
            yy = addComplexGaussianNoise(yy, variance=var)
            y[i,:] = yy 
            
        self.transmitted = y
        return y
        
    def receive(self, x=None):
        if(x is None):
            if(self.transmitted is None):
                raise ValueError('The sequence to be decoded must be either supplied or the transmit method of the class must be used before calling decode')
            else:
                x = self.transmitted
        
        s = x.shape
        if(s[0]%2 != 0):
            raise ValueError('The number of rows of the input must be divisible by two. (First row t = 0, second row t = 0 + Ts)')
        y = np.zeros(shape=(s[0], 1), dtype=complex)
        H = self.channel.getChannel()
        
        for i in range(s[0]//2):
            s1 = 0 + 0j
            s2 = 0 + 0j
            
            for j in range(self.Nr):
                s1 += np.conj(H[j,0])*x[2*i,j] + H[j,1]*np.conj(x[2*i+1,j])
                
                s2 += np.conj(H[j, 1])*x[2*i, j] - H[j, 0]*np.conj(x[2*i+1, j])
            
            y[2*i] = s1
            y[2*i+1] = s2
        
        y = y.flatten()
        self.decoded = y
        return y
    
    def decode(self, x=None):
        if(x is None):
            if(self.decoded is None):
                raise ValueError('Nothing to decode, a vector must be supplied or the receive method called before calling decode')
            else:
                x = self.decoded
        
        dictionary = self.generator.getDictionary()
        
        H = self.channel.getChannel()
        H = np.absolute(H)
        H = np.square(H)
        K = np.sum(H)
        
        y = np.zeros(len(x), dtype=complex)
    
        for i in range(len(x)):
            minnorm = math.inf
            symb = 0
            
            for j in range(len(dictionary)):
                v = np.absolute(x[i] - K*dictionary[j])**2
                if(v < minnorm):
                    minnorm = v
                    symb = dictionary[j]
            
            y[i] = symb
        
        y = y.reshape(-1,1)
        self.MLdecoding = y

        return y

    def Errors(self):
        self.errors = np.sum(np.real(self.originalMessage) != np.real(self.MLdecoding)) +  np.sum(np.imag(self.originalMessage) != np.imag(self.MLdecoding))
        self.error_rate = self.errors/(len(self.originalMessage)*self.generator.BitsPerSymbol())
        
        return self.errors, self.error_rate
    
    def SendAndReceive(self,x, variance=1, Power=None):
        x = x.reshape(-1,1)
        self.code(x)
        self.transmit(var=variance, power = Power)
        self.receive()
        self.decode()
        return self.Errors()
