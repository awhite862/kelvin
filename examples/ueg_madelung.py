import numpy
from kelvin.ueg_scf_system import ueg_scf_system
from kelvin.ueg_system import UEGSystem
from kelvin.ccsd import ccsd

T = 0.0
na = 7
nb = 7
N = na + nb
L = 3.88513
mu = 2.0
ueg = UEGSystem(T,L,9.4,mu=mu,norb=33,orbtype='u',madelung='const')
print('density: {}'.format(ueg.den))
print('r_s: {}'.format(ueg.rs))
cc = ccsd(ueg,T=T,mu=mu,iprint=1)
E = cc.run()
print(E)
ueg = UEGSystem(T,L,9.4,mu=mu,norb=33,orbtype='u',madelung='orb')
print('density: {}'.format(ueg.den))
print('r_s: {}'.format(ueg.rs))
cc = ccsd(ueg,T=T,mu=mu,iprint=1)
E = cc.run()
print(E)
