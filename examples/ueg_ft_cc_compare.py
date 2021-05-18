from kelvin.ccsd import ccsd
from kelvin.ueg_system import UEGSystem
from kelvin.ueg_scf_system import UEGSCFSystem
import numpy

T = 0.1
mu = 0.1182968
L = 2*numpy.pi/numpy.sqrt(1.0)
ng = 40
norb = 7
cut = 1.2
damp = 0.1
mi = 50
ueg = UEGSystem(T, L, cut, mu=mu, norb=norb, orbtype='u')
print('Norb: {}'.format(len(ueg.basis.basis)))
print('L: {:.10f}'.format(L))
print('N0: {:.10f}'.format(ueg.N))
print('mu: {:.10f}'.format(ueg.mu))
print('density: {:.10f}'.format(ueg.den))
print('r_s: {:.10f}'.format(ueg.rs))
print('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(
    ueg, T=T, mu=mu, iprint=1, max_iter=mi, damp=damp, ngrid=ng, tconv=1e-8)
Ecctot, Ecc = ccsdT.run()
print('Omega: {:.10f}'.format(Ecctot))
print('OmegaC: {:.12f}'.format(Ecc))

ccsdT.compute_ESN()
print("E: {}".format(ccsdT.E))
print("S: {}".format(ccsdT.S))
print("N: {}".format(ccsdT.N))

T = 0.1
mu = 0.1182817
L = 2*numpy.pi/numpy.sqrt(1.0)
norb = 7
cut = 1.2
damp = 0.1
mi = 50
ueg = UEGSCFSystem(T, L, cut, mu=mu, norb=norb, orbtype='u')
print('Norb: {}'.format(len(ueg.basis.basis)))
print('L: {:.10f}'.format(L))
print('N0: {:.10f}'.format(ueg.N))
print('mu: {:.10f}'.format(ueg.mu))
print('density: {:.10f}'.format(ueg.den))
print('r_s: {:.10f}'.format(ueg.rs))
print('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(
    ueg, T=T, mu=mu, iprint=1, max_iter=mi, damp=damp, ngrid=ng, tconv=1e-8)
Ecctot, Ecc = ccsdT.run()
print('Omega: {:.10f}'.format(Ecctot))
print('OmegaC: {:.12f}'.format(Ecc))

ccsdT.compute_ESN()
print("E: {}".format(ccsdT.E))
print("S: {}".format(ccsdT.S))
print("N: {}".format(ccsdT.N))
