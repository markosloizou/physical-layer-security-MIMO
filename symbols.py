import numbers
import numpy
import random
import math

class SymbolGenerator:
    def __init__(self, symbols='binary', p=0.5):
        if(not isinstance(p,numbers.Number)):
            raise ValueError('p must be a numbe')
        if(p < 0 or p > 1):
            raise ValueError('p must be between 0 and 1')
        
        if(symbols == 'binary'):
            pass
        elif(symbols == 'BPSK'):
            pass
        elif(symbols == 'QPSK'):
            symbols_dict = numpy.zeros(4,dtype=complex)
        
            symbols_dict[0] = complex(1,1)
            symbols_dict[1] = complex(-1, 1)
            symbols_dict[2] = complex(1,-1)
            symbols_dict[3] = complex(-1,-1)
            
            self.dictionary = symbols_dict
        elif(symbols == '8QAM'):
            symbols_dict = numpy.zeros(8,dtype=complex)
            symbols_dict[0] = complex(1,1)
            symbols_dict[1] = complex(-1, 1)
            symbols_dict[2] = complex(1,-1)
            symbols_dict[3] = complex(-1,-1)
            symbols_dict[4] = complex(1+math.sqrt(3),0)
            symbols_dict[5] = complex(0, 1 + math.sqrt(3))
            symbols_dict[6] = complex(-1-math.sqrt(3),0)
            symbols_dict[7] = complex(0,-1-math.sqrt(3))
            
            self.dictionary = symbols_dict
        elif (symbols == '16QAM'):
            pass
        elif (symbols == '32QAM'):
            pass
        elif(symbols == '64QAM'):
            pass
        elif(symbols == '128QAM'):
            pass
        else:
            raise ValueError('Symbol type not recognised or supported')
        
        self.symbols = symbols
        self.p = p
    
    def generate(self, n):
        if(not isinstance(n,int)):
            raise ValueError('n must be an integer')
        if(n<0):
            raise ValueError('n must be greater than zero')
        
        
        if(self.symbols == 'binary'):
            return self.binary(n)
        elif(self.symbols == 'BPSK'):
            return self.BPSK(n)
        elif(self.symbols == 'QPSK'):
            return self.QPSK(n)
        elif(self.symbols == '8QAM'):
            return self.QAM8(n)
        elif (self.symbols == '16QAM'):
            print('Not yet implemented')
            pass
        elif (self.symbols == '32QAM'):
            print('Not yet implemented')
            pass
        elif(self.symbols == '64QAM'):
            print('Not yet implemented')
            pass
        elif(self.symbols == '128QAM'):
            print('Not yet implemented')
            pass
        else:
            raise ValueError('Symbol type not recognised or supported')
    
    
    def toBinary(self,x):
        if(self.symbols == 'binary'):
            return x
        elif(self.symbols == 'BPSK'):
            return self.BPSKtoBinary(n)
        elif(self.symbols == 'QPSK'):
            return self.QPSK(n)
        elif(self.symbols == '8QAM'):
            return self.QAM8(n)
        elif (self.symbols == '16QAM'):
            print('Not yet implemented')
            pass
        elif (self.symbols == '32QAM'):
            print('Not yet implemented')
            pass
        elif(self.symbols == '64QAM'):
            print('Not yet implemented')
            pass
        elif(self.symbols == '128QAM'):
            print('Not yet implemented')
            pass
        else:
            raise ValueError('Symbol type not recognised or supported')
    
    
    def binary(self,n):
        symbols = np.array(2)
        sybols[0] = 0
        symbols[1] = 1
        y = numpy.zeros(n,dtype=complex)
        for i in range(n):
            if(random.random()<self.p):
                y[i] = 1
        return y
    
    def BPSK(self,n):
        sybols[0] = -1
        symbols[1] = 1
        y = numpy.ones(n,dtype=complex)
        for i in range(n):
            if(random.random()<self.p):
                y[i] = -1
        return y
    
    def BPSKtoBinary(self,x):
        y = numpy.ones(n)
        for i in range(n):
            if(y[i] == -1):
                y[i] = 0
    
    def QPSK(self,n):
        y = numpy.ones(n,dtype=complex)
        
        symbols = numpy.zeros(4,dtype=complex)
        
        symbols[0] = complex(1,1)
        symbols[1] = complex(-1, 1)
        symbols[2] = complex(1,-1)
        symbols[3] = complex(-1,-1)
        
        self.dictionary = symbols
        
        for i in range(n):
            b1 = 1
            b2 = 1
            
            if(random.random()<self.p):
                b1 = 0
            if(random.random()<self.p):
                b2 = 0
            if(b1 == 1 and b2 == 1):
                y[i] = complex(1,1)
            elif(b1 == 1 and b2 == 0):
                y[i] = complex(1, -1)
            elif(b1 == 0 and b2 == 1):
                y[i] = complex(-1,1)
            else:
                y[i] = complex(-1,-1)
        
        return y
    
    def QAM8(self,n):
        y = numpy.ones(n,dtype=complex)
        
        symbols = numpy.zeros(8,dtype=complex)
        
        symbols[0] = complex(1,1)
        symbols[1] = complex(-1, 1)
        symbols[2] = complex(1,-1)
        symbols[3] = complex(-1,-1)
        symbols[4] = complex(1+math.sqrt(3),0)
        symbols[5] = complex(0, 1 + math.sqrt(3))
        symbols[6] = complex(-1-math.sqrt(3),0)
        symbols[7] = complex(0,-1-math.sqrt(3))
        
        self.dictionary = symbols
        
        for i in range(n):
            b1 = 1
            b2 = 1
            b3 = 1
            if(random.random()<self.p):
                b1 = 0
            if(random.random()<self.p):
                b2 = 0
            if(random.random()<self.p):
                b3 = 0
            
            if(b1 == 1 and b2 == 1 and b3 == 0):
                y[i] = complex(1,1)
            elif(b1 == 1 and b2 == 0 and b3 == 0):
                y[i] = complex(1, -1)
            elif(b1 == 0 and b2 == 1 and b3 == 0):
                y[i] = complex(-1,1)
            elif(b1 == 0 and b2 == 0 and b3 == 0):
                y[i] = complex(-1,-1)
            elif(b1 == 1 and b2 == 1 and b3 == 1):
                y[i] = complex(1+math.sqrt(3),0)
            elif(b1 == 1 and b2 == 0 and b3 == 1):
                y[i] = complex(0, 1 + math.sqrt(3))
            elif(b1 == 0 and b2 == 1 and b3 == 1):
                y[i] = complex(-1-math.sqrt(3),0)
            elif(b1 == 0 and b2 == 0 and b3 == 1):
                y[i] = complex(0,-1-math.sqrt(3))
        
        return y
    
    def EnergyPerSymbol(self):
        E = sum(numpy.square(numpy.absolute(self.dictionary)))
        return E/len(self.dictionary)
    
    def EnergyPerBit(self):
        return self.EnergyPerSymbol() / numpy.log2(len(self.dictionary))
    
    def BitsPerSymbol(self):
        return numpy.log2(len(self.dictionary))
    
    def getDictionary(self):
        return self.dictionary

import numpy
import math

class MLdecoder:
    def __init__(self,symbols='binary'):
        if(symbols == 'binary'):
            pass
        elif(symbols == 'BPSK'):
            pass
        elif(symbols == 'QPSK'):
            pass
        elif(symbols == '8QAM'):
            pass
        elif (symbols == '16QAM'):
            pass
        elif (symbols == '32QAM'):
            pass
        elif(symbols == '64QAM'):
            pass
        elif(symbols == '128QAM'):
            pass
        else:
            raise ValueError('Symbol type not recognised or supported')
        
        self.symbols = symbols
    
    def decode(self,x):
        if(self.symbols == 'binary'):
            return self.binary(x)
        elif(self.symbols == 'BPSK'):
            print('BPSK decoding not implemented yet')
        elif(self.symbols == 'QPSK'):
            return self.QPSK(x)
        elif(self.symbols == '8QAM'):
            print('8QAM decoding not implemented yet')
        elif (self.symbols == '16QAM'):
            print('16QAM decoding not implemented yet')
        elif (self.symbols == '32QAM'):
            print('32QAM decoding not implemented yet')
        elif(self.symbols == '64QAM'):
            print('64QAM decoding not implemented yet')
        elif(self.symbols == '128QAM'):
            print('128QAM decoding not implemented yet')
        else:
            raise ValueError('Symbol type not recognised or supported')
    
    def binary(self,x):
        for i in range(len(x)):
            if(x[i] < 0.5):
                x[i] = 0
            else:
                x[i] = 1
    
    
    def QPSK(self,x):
        y = numpy.zeros(len(x),dtype=complex)
        symbols = numpy.zeros(4,dtype=complex)
        
        symbols[0] = complex(1,1)
        symbols[1] = complex(-1, 1)
        symbols[2] = complex(1,-1)
        symbols[3] = complex(-1,-1)
        
        for i in range(len(x)):
            theta = numpy.angle(x[i])
            
            if((theta >= 0) and (theta < math.pi/2)):
                y[i] = symbols[0]
            elif((theta >= math.pi/2) and (theta < math.pi)):
                y[i] = symbols[1]
            elif((theta < 0) and (theta >= -math.pi/2)):
                y[i] = symbols[2]
            else:
                y[i] = symbols[3]
        return y
