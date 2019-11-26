import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MakeReal(BaseEstimator, TransformerMixin):
    def __init__(self, RealPart = True, ImaginaryPart=True, magnitude = False, Phase = False):
        self.real = RealPart
        self.imag = ImaginaryPart
        self.magnitude = magnitude
        self.phase = Phase
    def fit(self,X, y = None):
        return self
    def transform(self,XX,y=None):
        X = np.array(XX)
        c = 0
        if(self.real == True):
            c += 1
        if(self.imag == True):
            c += 1
        if(self.magnitude == True):
            c += 1
        if(self.phase == True):
            c += 1
        
        s = X.shape
        X_transformed = np.zeros((s[0], c*s[1]))
        
        for i in range(s[1]):
            k = 0
            if(self.real == True):
                X_transformed[:,c*i + k] = np.real(X[:,i])
                k+= 1
            if(self.imag == True):
                X_transformed[:,c*i+k] = np.imag(X[:,i])
                k+= 1
            if(self.magnitude == True):
                X_transformed[:,c*i+k] = np.absolute(X[:,i])
                k += 1
            if(self.phase == True):
                X_transformed[:,c*i+k] = np.angle(X[:,i])
                k += 1
        
        return X_transformed
