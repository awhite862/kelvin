import numpy
from kelvin.ccsd import ccsd
from kelvin.hubbard_system import hubbard_system
from lattice.hubbard import Hubbard1D

L = 6
U = 1.0
T = 1.0
mu = 0.0
hub = Hubbard1D(L, 1.0, U, boundary='p')

Oa = numpy.zeros((L))
Ob = numpy.zeros((L))
for i in range(L):
    if i % 2 == 0:
        Oa[i] = 1.0
    else:
        Ob[i] = 1.0
Pa = numpy.einsum('i,j->ij', Oa, Oa)
Pb = numpy.einsum('i,j->ij', Ob, Ob)
sys = hubbard_system(T, hub, Pa, Pb, mu=mu)
cc = ccsd(sys, iprint=1, max_iter=80, econv=1e-11, T=T, mu=mu)
Eout, Ecc = cc.run()
print(Eout-Ecc, Ecc)
