# MIMO channel simulation with Eavesdropper

Various MIMO channels have been implemented (Line of Sight, Rayleigh fading, etc). 
The main analysis done was to compare the physical layer security provided by 
antenna selection (any subset of the transmit antennas) using ML algorithms
to the theoretical maximum security rate. 

The main conclusion is that as the ML algorithms can approach the theoretical curves
as the training data increase. This is a problem where synthetic data can be used
as the channels are random in nature, so by using random channels good results
can be achieved.
