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
c = calculateBERmodelAS(decoder, sg, snr, mlpc_oas, pl_oas, nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters, OAS = True, verbose = True)
ber_oas_mlpc = c.getBer()
outfile = open('./model_results/ber_oas_mlpc_Nt{}Nr{}'.format(Nt,Nr), 'wb')
pickle.dump(ber_oas_mlpc, outfile)
outfile.close()

random.seed(42)
c = calculateBERmodelAS(decoder, sg, snr, mlpc_sas, pl_sas, nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters, OAS = False, verbose = True)
ber_sas_mlpc = c.getBer()
outfile = open('./model_results/ber_sas_mlpc_Nt{}Nr{}'.format(Nt,Nr), 'wb')
pickle.dump(ber_sas_mlpc, outfile)
outfile.close()

random.seed(42)
c = calculateBERmodelAS(decoder, sg, snr, mlpc_lroas, pl_lroas, nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters, OAS = False, verbose = True)
ber_lroas_mlpc = c.getBer()
outfile = open('./model_results/ber_lroas_mlpc_Nt{}Nr{}'.format(Nt,Nr), 'wb')
pickle.dump(ber_lroas_mlpc, outfile)
outfile.close()



#plt.semilogy(snr,ber_oas_mlpc, 'bo--', label='OAS MLPC')
#plt.semilogy(snr, ber_oas_knn, 'r>-.', label='OAS KNN')
#plt.semilogy(snr, ber_oas_svm, 'gp:', label = 'OAS SVM')

#plt.semilogy(snr,ber_sas_mlpc, 'b*--', label = 'SAS MLPC')
#plt.semilogy(snr, ber_sas_knn, 'r^-.', label = 'SAS KNN')
#plt.semilogy(snr, ber_sas_svm, 'g<:', label = 'SAS SVM')

#plt.semilogy(snr,ber_lroas_mlpc, 'bx--', label = 'LROAS MLPC')
#plt.semilogy(snr, ber_lroas_knn, 'g+:', label = 'LROAS SVM')

#plt.legend()
#plt.show()
