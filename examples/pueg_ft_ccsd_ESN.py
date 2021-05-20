import logging
import sys as csys
from kelvin.ccsd import ccsd
from kelvin.pueg_system import PUEGSystem
import numpy

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    stream=csys.stdout)

T = 0.5
mu = 0.2
L = 2*numpy.pi/numpy.sqrt(1.0)
norb = 19
cut = 1.2
damp = 0.2
mi = 50
ueg = PUEGSystem(T, L, cut, mu=mu, norb=norb)
logging.info('Norb: {}'.format(len(ueg.basis.basis)))
logging.info('L: {:.10f}'.format(L))
logging.info('N0: {:.10f}'.format(ueg.N))
logging.info('mu: {:.10f}'.format(ueg.mu))
logging.info('density: {:.10f}'.format(ueg.den))
logging.info('r_s: {:.10f}'.format(ueg.rs))
logging.info('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(ueg, T=T, mu=mu, iprint=1, max_iter=mi, damp=damp, ngrid=10)
Ecctot, Ecc = ccsdT.run()
logging.info('Omega: {:.10f}'.format(Ecctot))
logging.info('OmegaC: {:.12f}'.format(Ecc))

ccsdT.compute_ESN()
logging.info("E: {}".format(ccsdT.E))
logging.info("S: {}".format(ccsdT.S))
logging.info("N: {}".format(ccsdT.N))
