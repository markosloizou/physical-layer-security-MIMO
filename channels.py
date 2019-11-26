import numbers
import random
import cmath
import math
import numpy as np
import scipy.linalg as sla


#Basic channel model with a channel matrix that can be set to subset(for antenna selection problems)
class Channel:
    
    def setSubchannel(self,s):
        self.ChannelMatrix = self.ChannelMatrix[:,s]
        
    def dataIn(self,x, variance = 0):
        s = self.ChannelMatrix.shape
        if(len(s) == 0):
            y = x*self.ChannelMatrix
        else:
            x = np.array(x)
            y = np.matmul(self.ChannelMatrix,x)
        return addComplexGaussianNoise2(y, variance = variance)
    
    def getChannel(self, variance = None):
        y = self.ChannelMatrix
        if(variance is not None):
            return addComplexGaussianNoise2(y, variance = variance)
        else:
            return y

class BinarySymmetricChannel(Channel):
    def __init__(self, p=0.5,dictionary=[0,1],seed=None):
        if(not isinstance(p, numbers.Number)):
            raise ValueError('p in BinarySymmetricChannel constructor must be a number between zero and one')
        if(p < 0 or p > 1):
            raise ValueError('p in BinarySymmetricChannel constructor must be between zero and one')
        self.p_error = p
        
        if(len(set(dictionary)) != 2):
            raise ValueError('The dictionary must consist only two and distinct values')
        self.d_list = dictionary
        
        if(seed != None):
            if(isinstance(seed, numbers.Number)):
                random.seed(seed)
            else:
                raise ValueError('The seed must be an integer ')
    
    def dataIn(self, x):
        self.n_errors = 0
        y = x
        for i in range(len(y)):
            if(random.random() < self.p_error):
                self.n_errors += 1
                if(y[i] == self.d_list[0]):
                    y[i] = self.d_list[1]
                else:
                    y[i] = self.d_list[0]
        self.error_rate = self.n_errors / len(x)
        return y
    

    
    def getChannel(self,variance=None):
        y = self.p_error
        if(variance is not None):
            y = y + np.random.normal(scale = variance)
        return y
    
    def numErrors(self):
        return self.n_errors
    
    def errorRate(self):
        return self.error_rate


class MIMOChannelLOS(Channel):
    def __init__(self, nt, nr, dt, dr, D, f):
        if(not isinstance(nt,int)):
            raise ValueError('nt(number of transmit antennas) must be an integer')
        if(not isinstance(nr,int)):
            raise ValueError('nr(number of receive antennas) must be an integer')
        if(not isinstance(dt,numbers.Number)):
            raise ValueError('dt(distance between transmit antennas) must be a number')
        if(not isinstance(dr,numbers.Number)):
            raise ValueError('dr(distance between receive antennas) must be a number')
        if(not isinstance(D,numbers.Number)):
            raise ValueError('D(distance between transmit and receive antenna arrays) must be a number')
        if(not isinstance(f,numbers.Number)):
            raise ValueError('f(frequency) must be a number')
        self.nt = nt
        self.nr = nr
        self.dt = dt
        self.dr = dr
        self.D = D
        self.f = f
        self.c = 299792458
        self.attenuation = (4*math.pi*f*D/self.c)**2
        self.wavelength = self.c/self.f
        self.common_phase_shift = 2*math.pi*D/self.wavelength
        
        H = np.zeros(shape=(nr,nt), dtype=complex)
        for i in range(nt):
            h = np.zeros(nr, dtype=complex)
            thetak = math.asin(i*self.dt/ self.D)
            for k in range(nr):
                h[k] = cmath.exp(-1j*(k)*2*math.pi*math.sin(thetak)*self.dt/self.wavelength)
            
            h = h*cmath.exp(-1j*self.common_phase_shift)/self.attenuation
            
            H[:,i] = h.T
        
        self.ChannelMatrix = H
        
    
    



class MIMOChannelUncorrelatedGaussian(Channel):
    def __init__(self,nt,nr,sigma=0.5):
        if(not isinstance(sigma,numbers.Number)):
            raise ValueError('sigma must be a number')
        
        H = np.zeros(shape=(nr,nt), dtype = complex)
        for i in range(nr):
            for j in range(nt):
                H[i,j] = complex(random.gauss(0,sigma),random.gauss(0,sigma))
        self.ChannelMatrix = H
    
    def getChannel(self):
        return self.ChannelMatrix
    
    def dataIn(self,x):
        x = np.array(x)
        return np.matmul(self.ChannelMatrix,x)
    
class GeneralCorrelatedMIMOChannel(Channel):
    def __init__(self, nr, nt, R,sigma=0.5):
        self.R = np.array(R,dtype=complex)
        s = self.R.shape
        if(not isinstance(nt,int)):
            raise ValueError('nt(number of transmit antennas) must be an integer')
        if(not isinstance(nr,int)):
            raise ValueError('nr(number of receive antennas) must be an integer')
        if(not isinstance(sigma,numbers.Number)):
            raise ValueError('sigma must be a number')
        if(len(s)!=2):
            raise ValueError('R must be a two dimensional Matrix')
        if(s[0] != s[1]):
            raise ValueError('R must be square with dimensions (NrNt)x(NrNt)')
        
        Hw = np.zeros(shape=(nr,nt), dtype = complex)
        for i in range(nr):
            for j in range(nt):
                Hw[i,j] = complex(random.gauss(0,sigma),random.gauss(0,sigma))
        H = np.matmul(sla.sqrtm(R),Hw.flatten('F'))
        H = np.array(H)
        self.ChannelMatrix = H.reshape(nr,nt)
    

    


class KroneckerRayleighChannelMIMO(Channel):
    def __init__(self,nt,nr,Rt,Rr, sigma=1, seed=None):
        if(not isinstance(nt,int)):
            raise ValueError('nt(number of transmit antennas) must be an integer')
        if(not isinstance(nr,int)):
            raise ValueError('nr(number of receive antennas) must be an integer')
        if(not isinstance(sigma,numbers.Number)):
            raise ValueError('sigma must be a number')
        
        if(seed is not None):
            random.seed(seed)
        Hw = np.zeros(shape=(nr,nt), dtype = complex)
        for i in range(nr):
            for j in range(nt):
                Hw[i,j] = complex(random.gauss(0,sigma),random.gauss(0,sigma))
        
        self.ChannelMatrix = np.matmul(np.matmul(sla.sqrtm(Rr),Hw), np.matrix(sla.sqrtm(Rt)).getH())
    

    


class PinholeChannelMIMO(Channel):
    def __init__(self,nt,nr,sc=1):
        if(not isinstance(nt,int)):
            raise ValueError('nt(number of transmit antennas) must be an integer')
        if(not isinstance(nr,int)):
            raise ValueError('nr(number of receive antennas) must be an integer')
        if(not isinstance(sc,numbers.Number)):
            raise ValueError('The scale(sc) of the rayleigh distribution must be a number')
            
        hr = np.random.normal(scale=sc,size=(nr,1)) + np.multiply(1j,np.random.normal(scale=sc,size=(nr,1)))
        ht = np.random.normal(scale=sc,size=(1,nt)) + np.multiply(1j,np.random.normal(scale=sc,size=(1,nt)))
        self.ChannelMatrix = hr*ht
        

    


class RayleighLOSChannelMIMO(Channel):
    def __init__(self,nt,nr,dt, dr, D, f,K):
        if(not isinstance(nt,int)):
            raise ValueError('nt(number of transmit antennas) must be an integer')
        if(not isinstance(nr,int)):
            raise ValueError('nr(number of receive antennas) must be an integer')
        if(not isinstance(dt,numbers.Number)):
            raise ValueError('dt(distance between transmit antennas) must be a number')
        if(not isinstance(dr,numbers.Number)):
            raise ValueError('dr(distance between receive antennas) must be a number')
        if(not isinstance(D,numbers.Number)):
            raise ValueError('D(distance between transmit and receive antenna arrays) must be a number')
        if(not isinstance(f,numbers.Number)):
            raise ValueError('f(frequency) must be a number')
        if(not isinstance(K,numbers.Number)):
            raise ValueError('K must be a number')
        
        chLOS = MIMOChannelLOS(nt,nr,dt,dr,D,f)
        HLOS = chLOS.getChannel()
        
        Hw = np.zeros(shape=(nr,nt), dtype = complex)
        for i in range(nr):
            for j in range(nt):
                Hw[i,j] = complex(random.gauss(0,0.5),random.gauss(0,0.5))
        
        self.ChannelMatrix = math.sqrt(K/(1+K))*HLOS + math.sqrt(1/(1+K))*Hw
    

    


def addComplexGaussianNoise(x,mu=0,variance=1):
    return np.array(x,dtype=complex) + np.random.normal(mu,variance,x.shape) + 1j*np.random.normal(mu,variance,x.shape)

def addRealGaussianNoise(x,mu=0,variance=1):
    return x + np.random.normal(mu,variance,x.shape)



import threading
import time

def addComplexGaussianNoise2(x,mu=0,variance=1):
    n1 = np.random.normal(loc=mu, scale=variance, size=x.shape)
    n2 = np.random.normal(loc=mu, scale=variance, size=x.shape)*1j
    n3 = n1 + n2
    y = x + n3
    return y

def calculateBER(channel, receiver, decoder, sGenerator ,snr, Nsymbols=1000, iterations=1000,verbose=False):
    Eb = sGenerator.EnergyPerBit()
    ber = np.zeros((len(snr)))
    total = len(snr)
    counter = 0
    previous = -1
    start1 = time.time()
    for n in snr:
        
        No = Eb/(10**(n/10))
        errRate = 0
        start = time.time()
        for i in range(iterations):
            s = sGenerator.generate(Nsymbols)
            s = s.reshape((1,-1))
            y = channel.dataIn(s)
            yn = addComplexGaussianNoise2(np.array(y, dtype = complex), variance = No)
            z = receiver.receive(yn)
            shat = decoder.decode(np.asarray(z).reshape(-1))
            errors = 0
            s = s.flatten()
            for j in range(Nsymbols):
                if(np.real(s[j]) != np.real(shat[j])):
                    errors += 1
                if(np.imag(s[j]) != np.imag(shat[j])):
                    errors += 1
            errorRate = errors/Nsymbols
            errRate += errorRate
        
        errRate = errRate/iterations
        
        ber[counter] = errRate
        
        if(previous != (100*counter)//total and verbose == True):
            end = time.time()
            print((100*counter)//total, '% done, time elapsed: ', round(end-start1),'s')
            previous = (100*counter)//total
        counter +=1
    
    if(verbose==True):
        print('Total time elapsed:', round(end-start1))
    return ber

def  calculateBERfast(channel, receiver, decoder, sGenerator ,snr, Nsymbols=1000, iterations=1000, verbose=False,single=True):
    ber = np.zeros((len(snr)))
    index = 0
    threads  = []
    print(len(snr), ' threads will be created')
    if(verbose == True):
        if(single == True):
            print('Only the first thread will report it\'s state')
        else:
            print('All threads will report their states')
    else:
        print('No information will be given about the current state of the process, this may take a long time.'
             'Set verbose to true for logging')
    for n in snr:
        thread = BERfastInternal(channel, receiver, decoder, sGenerator ,n, ber, index, Nsymbols, iterations,verbose)
        if(single==True):
            verbose = False
        threads.append(thread)
        thread.start()
        index += 1
    
    for t in threads:
        t.join()
    
    return ber


class BERfastInternal(threading.Thread):
    def __init__(self, channel, receiver, decoder, sGenerator ,snr, array, index, Nsymbols, iterations, verbose=False):
        threading.Thread.__init__(self)
        self.channel = channel
        self.receiver  = receiver
        self.decoder = decoder
        self.sGenerator = sGenerator 
        self.snr = snr
        self.array = array
        self.index= index
        self.Nsymbols = Nsymbols
        self.iterations = iterations
        self.verbose = verbose
    
    
    def run(self):
    
        Eb = self.sGenerator.EnergyPerBit()
        No = Eb/(10**(self.snr/10))
        errRate = 0
        total = self.iterations

        previous = -1
        start1 = time.time()

        for i in range(self.iterations):
            
            
            s = self.sGenerator.generate(self.Nsymbols)
            s = s.reshape((1,-1))
            y = self.channel.dataIn(s, variance = No)
            z = self.receiver.receive(y)
            z = z.flatten()
            z = np.asarray(z).reshape(-1)
            shat = self.decoder.decode(np.asarray(z).reshape(-1))
            errors = 0
            s = s.flatten()
            
            #count errors
            errors = np.count_nonzero(np.real(s)!=np.real(shat)) + np.count_nonzero(np.imag(s)!=np.imag(shat))
            
            errorRate = errors/(self.Nsymbols*self.sGenerator.BitsPerSymbol())
            errRate += errorRate
            
            if(previous != (100*i)//total and self.verbose == True):
                end = time.time()
                if((100*i)//total != 0):
                    print('Thread {:3}: {:3}% done, elapsed time: {}s, expected time to completion: {}s'.format(
                    self.index, (100*i)//total, round(end-start1), round((100-((100*i)//total))*(end-start)) ))
                else:
                    print('Thread {:3}: {:3}% done'.format(self.index,(100*i)//total))
                      
                previous = (100*i)//total
                start = time.time()
        
        errRate = errRate/self.iterations
        self.array[self.index] = errRate
        
        if(self.verbose==True):
            end = time.time()
            print('Thread ', self.index, ' finished, elapsed time: ', round(end - start1), 's')

def MIMOChannelCapacity(channel, noise_variance = 1, covariance = None, bandwidth=1, Complex = True):
    if(isinstance(channel, Channel)):
        H = channel.getChannel()
    else:
        H = channel
    
    if(Complex == True):
        if(covariance is None):
            M = np.matmul(H,H.getH())
        else:
            M = np.matmul(H,covariance)
            M = np.matmul(M,H.getH())
        
        s = M.shape
        if(len(s) == 1):
            I = np.identity(1)
        else:
            I = np.identity(s[0])
        
        Sm = I + np.division(M, noise_variance)
        C = 2*bandwidth*np.log2(np.linalg.det(Sm))
        
    else:
        if(covariance is None):
            M = np.matmul(H,H.T)
        else:
            M = np.matmul(H,covariance)
            M = np.matmul(M,H.T)
        
        s = M.shape
        if(len(s) == 1):
            I = np.identity(1)
        else:
            I = np.identity(s[0])
        
        Sm = I + np.division(M, noise_variance)
        C = bandwidth*np.log2(np.linalg.det(Sm))
        
