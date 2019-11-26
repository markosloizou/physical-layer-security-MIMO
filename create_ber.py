import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

from DDAS import MakeReal
from AntennaSelection import generateLROASdata, generateOASdata, generateSASdata, calculateBERAS, calculateBERAS
from symbols import SymbolGenerator, MLdecoder



Nt = 2
Nr = 10

symbs=10000
iters=1000

sg = SymbolGenerator('QPSK')
decoder = MLdecoder('QPSK') 

snr = np.linspace(-5,10,31)
outfile = open('ber_snr', 'wb')
pickle.dump(snr, outfile)
outfile.close()

random.seed(42)
c = calculateBERAS(decoder, sg,snr, selection='SAS', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_sas = c.getBer()
outfile = open('ber_sas', 'wb')
pickle.dump(BER_sas,outfile)
outfile.close()

random.seed(42)
c = calculateBERAS(decoder, sg,snr, selection='OAS', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_oas = c.getBer()
outfile = open('ber_oas', 'wb')
pickle.dump(BER_oas,outfile)
outfile.close()

random.seed(42)
c = calculateBERAS(decoder, sg,snr, selection='LROAS', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_lroas = BER_oas = c.getBer()
outfile = open('ber_lroas', 'wb')
pickle.dump(BER_lroas,outfile)
outfile.close()

random.seed(42)
c =calculateBERAS(decoder, sg,snr, selection='None', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_none = c.getBer() 
outfile = open('ber_all', 'wb')
pickle.dump(BER_none,outfile)
outfile.close()

random.seed(42)
c =calculateBERAS(decoder, sg,snr, selection='random single', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_random_single = c.getBer()
outfile = open('ber_random_single', 'wb')
pickle.dump(BER_random_single,outfile)
outfile.close()

random.seed(42)
c = calculateBERAS(decoder, sg,snr, selection='random subset', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_random_subset = c.getBer()
outfile = open('ber_random_subset', 'wb')
pickle.dump(BER_random_subset,outfile)
outfile.close()

random.seed(42)
c = calculateBERAS(decoder, sg,snr, selection='MaxMinNorm', nt = Nt, nr = Nr, Nsymbols = symbs, iterations = iters,verbose=True)
BER_MaxMinNorm = c.getBer() 
outfile = open('ber_MaxMinNorm', 'wb')
pickle.dump(BER_MaxMinNorm,outfile)
outfile.close()

plt.figure()
plt.semilogy(snr, BER_none, 'ko--', label='All Antennae')

plt.semilogy(snr, BER_oas, 'ro--', label='OAS')
plt.semilogy(snr, BER_sas, 'r^-.', label='SAS')
plt.semilogy(snr, BER_lroas, 'r*:', label='LROAS')

plt.semilogy(snr, BER_MaxMinNorm, 'bo--', label='Max Min Norm')

plt.semilogy(snr, BER_random_single, 'go--', label='Random Single')
plt.semilogy(snr, BER_random_subset, 'g^-.', label='Random Multi')

plt.legend()
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')
plt.show()
