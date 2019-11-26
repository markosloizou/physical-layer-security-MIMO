import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

from DDAS import MakeReal
from AntennaSelection import generateLROASdata, generateOASdata, generateSASdata, calculateBERwithAS
from symbols import SymbolGenerator, MLdecoder


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from BERcalculation import calculateBERmodelAS

Nt = 2
Nr = 10

symbs=10000
iters=1000

sg = SymbolGenerator('QPSK')
decoder = MLdecoder('QPSK') 
snr = np.linspace(-5,10,31)


infile = open('./models/pl_oas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
pl_oas =  pickle.load(infile)
infile.close()

infile = open('./models/pl_sas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
pl_sas =  pickle.load(infile)
infile.close()

infile = open('./models/pl_lroas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
pl_lroas =  pickle.load(infile)
infile.close()

outfile = open('ber_snr', 'wb')
pickle.dump(snr, outfile)
outfile.close()


infile = open('./models/mlpc_oas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
mlpc_oas = pickle.load(infile)
infile.close()

infile = open('./models/mlpc_sas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
mlpc_sas = pickle.load(infile)
infile.close()

infile = open('./models/mlpc_lroas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
mlpc_lroas = pickle.load(infile)
infile.close()

infile = open('./models/knn_oas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
knn_oas = pickle.load(infile)
infile.close()

infile = open('./models/knn_sas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
knn_sas = pickle.load(infile)
infile.close()

infile = open('./models/knn_lroas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
knn_lroas = pickle.load(infile)
infile.close()

infile = open('./models/svm_oas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
svm_oas = pickle.load(infile)
infile.close()

infile = open('./models/svm_sas_Nt{}Nr{}'.format(Nt,Nr), 'rb')
svm_sas = pickle.load(infile)
infile.close()


random.seed(42)
c = calculateBERmodelAS(decoder, sg, snr, mlpc_lroas, pl_lroas, nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters, OAS = False, verbose = True)
ber_lroas_mlpc = c.getBer()
outfile = open('ber_lroas_mlpc_Nt{}Nr{}'.format(Nt,Nr), 'wb')
pickle.dump(ber_lroas_mlpc, outfile)
outfile.close()



plt.figure()
plt.semilogy(snr, ber_lroas_mlpc, label='MLPC')
plt.semilogy(snr, ber_lroas_knn, label='KNN')
plt.legend()
plt.show()
