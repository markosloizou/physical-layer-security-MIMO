import itertools
import numpy as np
import random
import threading
import time
import numbers

from channels import Channel, KroneckerRayleighChannelMIMO
from coding import AlamoutiCoding
from receivers import MMRC


class calculateBERmodelAS:
    def __init__(self,decoder, sGenerator ,snr, model, pipeline, nt=1, nr=1, Rr=None, Rt = None, Nsymbols=1000, iterations=1000, norm = 'fro', verbose=False,single=False, OAS = False):
        self.OAS = OAS
        self.model = model
        self.pipeline = pipeline
        self.decoder = decoder
        self.sGenerator = sGenerator 
        self.snr = snr
        self.Nsymbols = Nsymbols
        self.iterations = iterations
        self.verbose = verbose
        self.norm = norm
        self.nt = nt
        self.nr = nr
        self.Rr = Rr
        self.Rt = Rt
        
        if(Rr is None):
            self.Rr = np.identity(nr, dtype=float)
        if(Rt is None):
            self.Rt = np.identity(nt, dtype=float)
        
        
        self.ber = np.zeros((len(snr)))
        self.index = 0
        
        for n in snr:
            self.ber[self.index] = self.run(n)
            if(single==True):
                self.verbose = False
            self.index += 1
    
    def getBer(self):
        return self.ber
    
    def run(self, n):
    
        Eb = self.sGenerator.EnergyPerBit()
        No = Eb/(10**(n/10))
        errRate = 0
        total = self.iterations

        previous = -1
        start1 = time.time()

        for i in range(self.iterations):
            
            
            x = self.sGenerator.generate(self.Nsymbols)
            x = x.reshape((1,-1))
            self.channel = KroneckerRayleighChannelMIMO(self.nt,self.nr, self.Rt, self.Rr)
            H1 = self.channel.getChannel().flatten()
            H1 = H1.reshape(1,-1)
            
            if(self.OAS == True):
                self.channelEve = KroneckerRayleighChannelMIMO(self.nt,self.nr, self.Rt, self.Rr)
                H2 = self.channelEve.getChannel().flatten()
                H2 = H2.reshape(1,-1)
                X = np.zeros(self.nt*self.nr*2, dtype = complex)
                X[:self.nt*self.nr] = H1
                X[self.nt*self.nr:] = H2
            else:
                self.channelEve = None
                X = H1
            
            X = X.reshape(1,-1)
            X = self.pipeline.transform(X)
            
            s = self.model.predict(X)
            
            if(isinstance(s, numbers.Number)):
                s = np.array(s, dtype = int)
            else:
                s = s.astype(int)
                ss = s.flatten()
                s = []
                for j in ss:
                    if(j==0):
                        continue
                    else:
                        s.append(j)
                s = np.array(s, dtype=int)
                s = s.flatten()

            self.channel.setSubchannel(s)
            
            if(self.OAS == True):
                self.channelEve.setSubchannel(s)
            
            if(len(s) == 1):
                self.receiver = MMRC(self.nr, self.channel)
            elif(len(s)==2):
                self.receiver = AlamoutiCoding(self.nr, self.channel, self.sGenerator)
            
            
            errors = 0
            errorRate = 0
            
            if(len(s) == 1):
                y = self.channel.dataIn(x, variance = No)
                z = self.receiver.receive(y)
                z = z.flatten()
                z = np.asarray(z).reshape(-1)
                shat = self.decoder.decode(np.asarray(z).reshape(-1))
                x = x.flatten()
                
                #count errors
                errors = np.count_nonzero(np.real(x)!=np.real(shat)) + np.count_nonzero(np.imag(x)!=np.imag(shat))
                errorRate = errors/(self.Nsymbols*self.sGenerator.BitsPerSymbol())
            
            elif(len(s) == 2):
                self.receiver.SendAndReceive(x, variance = No)
                errorRate = self.receiver.error_rate
            
        
            errRate += errorRate
            
            if(previous != (100*i)//total and self.verbose == True):
                end = time.time()
                if((100*i)//total != 0 and i != 0):
                    print('SNR {:3} of {:3}: {:3}% done, elapsed time: {:4}s, expected time to completion: {:4}s'.format(
                    self.index + 1,len(self.snr), (100*i)//total, round(end-start1), round((100-((100*i)//total))*(end-start)) ))
                else:
                    print('SNR {:3} of {:3}: {:3}% done'.format(self.index+1, len(self.snr), (100*i)//total))
                      
                previous = (100*i)//total
                start = time.time()
        
        errRate = errRate/self.iterations
        
        if(self.verbose==True):
            end = time.time()
            print('SNR {:3} of {:3}: finished.'.format(self.index+1, len(self.snr)))
        
        return errRate
