from kelvin.ccsd import ccsd
from kelvin.ueg_system import ueg_system
import time

T = 0.5
mu = 7.0
L = 1.942
norb = 33
cut = 30.0
damp = 0.0
mi = 50
ti = time.time()
ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='u')
print('Norb: {}'.format(len(ueg.basis.basis)))
print('L: {:.10f}'.format(L))
print('N0: {:.10f}'.format(ueg.N))
print('mu: {:.10f}'.format(ueg.mu))
print('density: {:.10f}'.format(ueg.den))
print('r_s: {:.10f}'.format(ueg.rs))
print('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(ueg,T=T,mu=mu,iprint=1,max_iter=mi,damp=damp,ngrid=10)
Ecctot,Ecc = ccsdT.run()
print('Omega: {:.10f}'.format(Ecctot))
print('OmegaC: {:.12f}'.format(Ecc))

ccsdT.compute_ESN()
print("E: {}".format(ccsdT.E))
print("S: {}".format(ccsdT.S))
print("N: {}".format(ccsdT.N))
tf = time.time()
print("Total time: {} s".format(tf - ti))
