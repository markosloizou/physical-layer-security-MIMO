import itertools
import numpy as np
import random
import threading
import time

from channels import Channel, KroneckerRayleighChannelMIMO, addComplexGaussianNoise2
from coding import AlamoutiCoding
from receivers import MMRC

class AntennaSelector:
    def __init__(self, userCh, eveCh=None):
        if(isinstance(userCh,Channel)):
            userCh = userCh.getChannel()
        if(len(userCh.shape)!= 2):
            raise ValueError('The user channel matrix must be a 2 dimensional array(even if it\'s a vector)')
        if(eveCh is not None):
            if(isinstance(eveCh,Channel)):
                eveCh = eveCh.getChannel()
            if(len(eveCh.shape)!= 2):
                raise ValueError('The eveasdropper channel matrix must be a 2 dimensional array(even if it\'s a vector)')
            if(eveCh.shape[0] != userCh.shape[0]):
                raise ValueError('The first dimension of the channels matrices must be the same,', 
                                 'ie the number of transmit antenas must be constant')
        self.userCh = userCh
        self.eveCh = eveCh
    
    def OptimalAntennaSelection(self, P=1, T=300, B=20e6, noise_variance=None, CSIvar=None):
        if(self.eveCh is None):
            raise ValueError('For optimal antenna selection both the user\'s and eavesdropper\'s channels are needed')
        
        if(CSIvar is not None):
            tempU = self.userCh
            tempE = self.eveCh
            self.userCh = addComplexGaussianNoise2(tempU, variance = CSIvar)
            self.eveCh = addComplexGaussianNoise2(tempE, variance = CSIvar)
        su = self.userCh.shape
        se = self.eveCh.shape
        
        self.noise_variance = 4*1.38e-23*B*T #calculate thermal noise variance
        
        if(noise_variance != None):
            if(not isinstance(noise_variance, numbers.Number)):
                raise ValueError('The noise variance must be a number')
            if(nosie_variance < 0):
                raise ValueError('The noise variance must be greater than or equal to zero')
            else:
                self.noise_variance = noise_variance
        
        bmax = 0
        imax= -1
        for i in range(su[1]):
            cu = self.userCh[:,i] # Get the ith column of each channel
            ce = self.eveCh[:,i]
            
            cu = np.absolute(cu) #calculate absolute value square of each entry
            cu = np.square(cu)
            ce = np.absolute(ce)
            ce = np.square(ce)
            
            r = (1 + P*sum(cu)/self.noise_variance)/(1 + P*sum(ce)/self.noise_variance)
            
            if(r > bmax):
                bmax = r
                imax = i
        if(CSIvar is not None):
            self.userCh = tempU
            self.eveCh = tempE
        return imax
    
    def SuboptimalAntennaSelection(self,CSIvar = None):
        if(CSIvar is not None):
            tempCh = self.userCh
            self.userCh = addComplexGaussianNoise2(tempCh, variance = CSIvar)
        
        s = self.userCh.shape
        maxi = 0;
        Ch = np.absolute(self.userCh) # Get the absolute value of each element
        Ch = np.square(Ch) #element wise square 
        sumCh = np.sum(Ch, axis = 0) #sum along columns
        
        if(CSIvar is not None):
            self.userCh = tempCh
        return np.argmax(sumCh) #return index of column with the greatest sum
    
    def LinearReceiverOptimalAntennaSet(self, CSIvar = None):
        #create set of antennas
        if(CSIvar is not None):
            temp = self.userCh
            self.userCh = addComplexGaussianNoise2(temp, variance = CSIvar)
        
        s = self.userCh.shape
        indices = np.array(range(0,s[1]))
        
        #create power set
        pset = self.powerset(indices)
        
        #evaluate minimum singular value of all submatrices
        minSingularValue = 0
        arrayIndexSet = np.array([])
        for x in pset:
            
            if (not x): #if array is empty continue(which will be at least once)
                continue
            
            #create new channel sub-matrix
            Unew = self.userCh[:,x]
            
            try:
                u, s, vh = np.linalg.svd(Unew)
            except LinAlgError as e:
                print(str(e))
            
            #chose the one with the maximum min singular value 
            if(np.amin(s) > minSingularValue):
                minSingularValue = np.amin(s)
                arrayIndexSet = x
        
        if(CSIvar is not None):
            self.userCh = temp
        return arrayIndexSet
    
    def MaxMinNorm(self, norm='fro'):
        
        #create set of antennas
        s = self.userCh.shape
        indices = np.array(range(0,s[1]))
        
        #create power set
        pset = self.powerset(indices)
        
        #evaluate norm of all submatrices
        minnorm = 0
        arrayIndexSet = np.array([])
        
        for x in pset:
            
            if (not x): #if array is empty continue(which will be at least once)
                continue
            
            #create new channel sub-matrix
            Unew = self.userCh[:,x]
            
            n = np.linalg.norm(Unew, ord=norm)
            
            #chose the one with the maximum min singular value 
            if(n > minnorm):
                minnorm = n
                arrayIndexSet = x
        
        return arrayIndexSet
    
    def RandomAntennaSelection(self, single = True):
        s = self.userCh.shape
        
        if(single == True):
            return  random.randint(0, s[1]-1)
        else:
            #create set of antennas
            indices = np.array(range(0,s[1]))
            
            #create power set
            pset = self.powerset(indices)
            
            s = []
            while(len(s) == 0):
                l = list(pset)
                s = l[random.randint(0, len(l)-1)]
                
            return s
            
        

    def powerset(self,L):
        pset = set()
        for n in range(len(L) + 1):
            for sset in itertools.combinations(L, n):
                pset.add(sset)
        return pset

def generateOASdata(n,nt=1,nr=1,Rt=None,Rr=None, CSIvariance = None, returnChannel = False):
    
    X = np.zeros((n,nt*nr*2), dtype=complex)
    y = np.zeros(n)
    userChannels = np.zeros(n, dtype=object)
    evesChannels = np.zeros(n, dtype=object)
    
    if(Rr is None):
        Rr = np.identity(nr)
    if(Rt is None):
        Rt = np.identity(nt)
    for i in range(n):
        chU =  KroneckerRayleighChannelMIMO(nt,nr,Rt,Rr)
        chEve = KroneckerRayleighChannelMIMO(nt,nr,Rt,Rr)
        AS = AntennaSelector(chU,chEve)
        if(CSIvariance is not None):
            H1 = chU.getChannel(variance = CSIvariance).flatten()
            H2 = chEve.getChannel(variance = CSIvariance).flatten()
        else:
            H1 = chU.getChannel().flatten()
            H2 = chEve.getChannel().flatten()
        H1 = H1.reshape(1,-1)
        H2 = H2.reshape(1,-1)
        X[i,:nt*nr] = H1
        X[i,nt*nr:] = H2
        y[i] = AS.OptimalAntennaSelection()
        userChannels[i] = chU
        evesChannels[i] = chEve
    if(returnChannel == False):
        return X,y
    else:
        return X,y, userChannels, evesChannels

def generateSASdata(n,nt=1,nr=1,Rt=None,Rr=None, CSIvariance = None, returnChannel = False):
    X = np.zeros((n,nt*nr), dtype=complex)
    y = np.zeros(n)
    userChannels = np.zeros(n, dtype=object)
    
    if(Rr is None):
        Rr = np.identity(nr)
    if(Rt is None):
        Rt = np.identity(nt)
    for i in range(n):
        chU =  KroneckerRayleighChannelMIMO(nt,nr,Rt,Rr)

        AS = AntennaSelector(chU)
        if(CSIvariance is not None):
            H1 = chU.getChannel(variance = CSIvariance).flatten()
        else:
            H1 = chU.getChannel().flatten()

        H1 = H1.reshape(1,-1)
        
        X[i,:] = H1
        y[i] = AS.SuboptimalAntennaSelection()
        userChannels[i] = chU
    
    if(returnChannel == False):
        return X,y
    else:
        return X,y,userChannels

def generateLROASdata(n,nt=1,nr=1,Rt=None,Rr=None, CSIvariance = None, returnChannel = False):
    X = np.zeros((n,nt*nr), dtype=complex)
    y = np.zeros((n,nt))
    userChannels = np.zeros(n, dtype=object)
    
    if(Rr is None):
        Rr = np.identity(nr)
    if(Rt is None):
        Rt = np.identity(nt)
    for i in range(n):
        chU =  KroneckerRayleighChannelMIMO(nt,nr,Rt,Rr)

        AS = AntennaSelector(chU)
        if(CSIvariance is not None):
            H1 = chU.getChannel(variance = CSIvariance).flatten()
        else:
            H1 = chU.getChannel().flatten()

        H1 = H1.reshape(1,-1)
        X[i,:] = H1

        S = AS.LinearReceiverOptimalAntennaSet()
        for s in S:
            y[i,s] = 1
        userChannels[i] = chU
        
    if(returnChannel == False):
        return X,y
    else:
        return X,y,userChannels


def  calculateBERwithAS(decoder, sGenerator ,snr,selection='OAS', nt=1, nr=1, Rr=None, Rt = None, Nsymbols=1000, iterations=1000, norm = 'fro', verbose=False,single=False):
    if(selection=='OAS'):
        pass
    elif(selection=='SAS'):
        pass
    elif(selection=='LROAS'):
        pass
    elif(selection=='random single'):
        pass
    elif(selection=='random subset'):
        pass
    elif(selection=='MaxMinNorm'):
        pass
    elif(selection=='None'):
        pass
    else:
        raise ValueError("{} is an invalid Antenna Selection method".format(selection))
    
    if(Rr is None):
        Rr = np.identity(nr)
    if(Rt is None):
        Rt = np.identity(nt)
    
    snr = np.array(snr)
    
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
        thread = BERfastInternalAS(selection,decoder, sGenerator ,n, ber, index, Nsymbols, iterations,norm, nt, nr, Rt, Rr, verbose)
        if(single==True):
            verbose = False
        threads.append(thread)
        thread.start()
        index += 1
    
    for t in threads:
        t.join()
    
    return ber


class BERfastInternalAS(threading.Thread):
    def __init__(self, selection, decoder, sGenerator ,snr, array, index, Nsymbols, iterations, norm, nt, nr, Rt, Rr, verbose=False):
        threading.Thread.__init__(self)
        self.selection = selection
        self.decoder = decoder
        self.sGenerator = sGenerator 
        self.snr = snr
        self.array = array
        self.index= index
        self.Nsymbols = Nsymbols
        self.iterations = iterations
        self.verbose = verbose
        self.norm = norm
        self.nt = nt
        self.nr = nr
        self.Rr = Rr
        self.Rt = Rt
    
    def run(self):
    
        Eb = self.sGenerator.EnergyPerBit()
        No = Eb/(10**(self.snr/10))
        errRate = 0
        total = self.iterations

        previous = -1
        start1 = time.time()

        for i in range(self.iterations):
            
            
            x = self.sGenerator.generate(self.Nsymbols)
            x = x.reshape((1,-1))
            self.channel = KroneckerRayleighChannelMIMO(self.nt,self.nr, self.Rt, self.Rr)
            if(self.selection == 'OAS'):
                self.channelEve = KroneckerRayleighChannelMIMO(self.nt,self.nr, self.Rt, self.Rr)
            else:
                self.channelEve = None
            
            AS = AntennaSelector(self.channel, self.channelEve)
            
            if(self.selection == 'OAS'):
                s = AS.OptimalAntennaSelection()
            elif(self.selection == 'SAS'):
                s = AS.SuboptimalAntennaSelection()
            elif(self.selection == 'LROAS'):
                s = AS.LinearReceiverOptimalAntennaSet()
            elif(self.selection == 'random single'):
                s = AS.RandomAntennaSelection(single=True)
            elif(self.selection == 'random subset'):
                s = AS.RandomAntennaSelection(single=False)
            elif(self.selection == 'MaxMinNorm'):
                s = AS.MaxMinNorm()
            elif(self.selection == 'None'):
                s = np.linspace(0, self.nt)
            else:
                raise ValueError("{} is not a valid antenna selection method.\nAvailable options are: OAS,SAS, LROAS, random(single or sybset) and Maximum minimum norm".format(self.selection))
            
            
            self.channel.setSubchannel(s)
            if(self.channelEve  is not None):
                self.channelEve.setSubchannel(s)
            
            if(isinstance(s,set)):
                if(len(s) == 1):
                    self.receiver = MMRC(self.nr, self.channel)
                if(self.selection=='LROAS' and np.sum(s) == 1):
                    self.receiver = MMRC(self.nr, self.channel)
                elif(np.sum(s) == 2):
                    self.receiver = AlamoutiCoding(self.nr, self.channel, self.sGenerator)
            else:
                self.receiver = MMRC(self.nr, self.channel)
            
            errors = 0
            errorRate = 0
            if(isinstance(s,set)):
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
                    self.receiver.SendAndReceive(x)
                    errorRate = self.receiver.error_rate
            else:
                y = self.channel.dataIn(x, variance = No)
                z = self.receiver.receive(y)
                z = z.flatten()
                z = np.asarray(z).reshape(-1)
                shat = self.decoder.decode(np.asarray(z).reshape(-1))
                x = x.flatten()
                
                #count errors
                errors = np.count_nonzero(np.real(x)!=np.real(shat)) + np.count_nonzero(np.imag(x)!=np.imag(shat))
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

class calculateBERAS:
    def __init__(self,decoder, sGenerator ,snr,selection='OAS', nt=1, nr=1, Rr=None, Rt = None, Nsymbols=1000, iterations=1000, norm = 'fro', verbose=False,single=False):
        self.selection = selection
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
        
        if(selection=='OAS'):
            pass
        elif(selection=='SAS'):
            pass
        elif(selection=='LROAS'):
            pass
        elif(selection=='random single'):
            pass
        elif(selection=='random subset'):
            pass
        elif(selection=='MaxMinNorm'):
            pass
        elif(selection=='None'):
            pass
        else:
            raise ValueError("{} is an invalid Antenna Selection method".format(selection))
        
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
            if(self.selection == 'OAS'):
                self.channelEve = KroneckerRayleighChannelMIMO(self.nt,self.nr, self.Rt, self.Rr)
            else:
                self.channelEve = None
            
            AS = AntennaSelector(self.channel, self.channelEve)
            
            if(self.selection == 'OAS'):
                s = AS.OptimalAntennaSelection()
            elif(self.selection == 'SAS'):
                s = AS.SuboptimalAntennaSelection()
            elif(self.selection == 'LROAS'):
                s = AS.LinearReceiverOptimalAntennaSet()
            elif(self.selection == 'random single'):
                s = AS.RandomAntennaSelection(single=True)
            elif(self.selection == 'random subset'):
                s = AS.RandomAntennaSelection(single=False)
            elif(self.selection == 'MaxMinNorm'):
                s = AS.MaxMinNorm()
            elif(self.selection == 'None'):
                s = np.linspace(0, self.nt-1, dtype=int)
            else:
                raise ValueError("{} is not a valid antenna selection method.\nAvailable options are: OAS,SAS, LROAS, random(single or sybset) and Maximum minimum norm".format(self.selection))
            
            
            if(self.selection != 'None'):
                self.channel.setSubchannel(s)
            if(self.channelEve  is not None):
                self.channelEve.setSubchannel(s)
            
            if(isinstance(s,set) or isinstance(s,tuple)):
                if(len(s) == 1):
                    self.receiver = MMRC(self.nr, self.channel)
                if(self.selection=='LROAS' and np.sum(s) == 1):
                    self.receiver = MMRC(self.nr, self.channel)
                elif(np.sum(s) == 2 or len(s) == 2):
                    self.receiver = AlamoutiCoding(self.nr, self.channel, self.sGenerator)
            
            elif(self.selection == 'None' and self.nt == 2):
                self.receiver = AlamoutiCoding(self.nr, self.channel, self.sGenerator)
            
            else:
                self.receiver = MMRC(self.nr, self.channel)
            
            
            errors = 0
            errorRate = 0
            if(isinstance(s,set) or isinstance(s,tuple)):
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
            elif(self.selection == 'None' and self.nt == 2):
                self.receiver.SendAndReceive(x, variance = No)
                errorRate = self.receiver.error_rate
            else:
                y = self.channel.dataIn(x, variance = No)
                z = self.receiver.receive(y)
                z = z.flatten()
                z = np.asarray(z).reshape(-1)
                shat = self.decoder.decode(np.asarray(z).reshape(-1))
                x = x.flatten()
                
                #count errors
                errors = np.count_nonzero(np.real(x)!=np.real(shat)) + np.count_nonzero(np.imag(x)!=np.imag(shat))
                errorRate = errors/(self.Nsymbols*self.sGenerator.BitsPerSymbol())
        
            errRate += errorRate
            
            if(previous != (100*i)//total and self.verbose == True):
                end = time.time()
                if((100*i)//total != 0):
                    print('SNR {:3} of {:3}: {:3}% done, elapsed time: {:4}s, expected time to completion: {:4}s'.format(
                    self.index + 1,len(self.snr), (100*i)//total, round(end-start1), round((100-((100*i)//total))*(end-start)) ))
                else:
                    print('SNR {:3} of {:3}: {:3}% done'.format(self.index+1, len(self.snr), (100*i)//total))
                      
                previous = (100*i)//total
                start = time.time()
        
        errRate = errRate/self.iterations
        
        if(self.verbose==True):
            end = time.time()
            print('SNR {:3} of {:3}: finished. Method = {}'.format(self.index+1, len(self.snr), self.selection))
        
        return errRate
