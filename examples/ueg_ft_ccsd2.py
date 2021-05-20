import logging
import sys as csys
from kelvin.ccsd import ccsd
from kelvin.ueg_system import UEGSystem
import numpy

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    stream=csys.stdout)

T = 0.1
mu = 0.1
L = 2*numpy.pi/numpy.sqrt(1.0)
norb = 19
cut = 1.2
damp = 0.2
mi = 50
ueg = UEGSystem(T, L, cut, mu=mu, norb=norb, orbtype='u')
logging.info('Norb: {}'.format(len(ueg.basis.basis)))
logging.info('L: {:.10f}'.format(L))
logging.info('N0: {:.10f}'.format(ueg.N))
logging.info('mu: {:.10f}'.format(ueg.mu))
logging.info('density: {:.10f}'.format(ueg.den))
logging.info('r_s: {:.10f}'.format(ueg.rs))
logging.info('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(ueg, T=T, mu=mu, iprint=1, max_iter=mi, damp=damp, ngrid=10)
Ecctot, Ecc = ccsdT.run()
T1 = ccsdT.T1
T2 = ccsdT.T2
logging.info('Omega: {:.10f}'.format(Ecctot))
logging.info('OmegaC: {:.12f}'.format(Ecc))
