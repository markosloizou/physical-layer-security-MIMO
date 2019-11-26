import numpy as np

class Receiver:
    def receive(self):
        pass

class MMRC(Receiver):
    def __init__(self, nr, channel):
        self.Channel = channel
        self.ChannelMatrix = channel.getChannel()
    
    def receive(self, x):
        self.s = np.multiply(x,np.conj(self.ChannelMatrix))
        self.ssum = sum(self.s)
        return self.ssum
