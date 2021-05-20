import logging
import sys as csys
from kelvin.ccsd import ccsd
from kelvin.ueg_system import UEGSystem
from kelvin.ueg_scf_system import UEGSCFSystem
import numpy

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    stream=csys.stdout)

T = 0.1
mu = 0.1182968
L = 2*numpy.pi/numpy.sqrt(1.0)
ng = 40
norb = 7
cut = 1.2
damp = 0.1
mi = 50
ueg = UEGSystem(T, L, cut, mu=mu, norb=norb, orbtype='u')
logging.info('Norb: {}'.format(len(ueg.basis.basis)))
logging.info('L: {:.10f}'.format(L))
logging.info('N0: {:.10f}'.format(ueg.N))
logging.info('mu: {:.10f}'.format(ueg.mu))
logging.info('density: {:.10f}'.format(ueg.den))
logging.info('r_s: {:.10f}'.format(ueg.rs))
logging.info('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(
    ueg, T=T, mu=mu, iprint=1, max_iter=mi, damp=damp, ngrid=ng, tconv=1e-8)
Ecctot, Ecc = ccsdT.run()
logging.info('Omega: {:.10f}'.format(Ecctot))
logging.info('OmegaC: {:.12f}'.format(Ecc))

ccsdT.compute_ESN()
logging.info("E: {}".format(ccsdT.E))
logging.info("S: {}".format(ccsdT.S))
logging.info("N: {}".format(ccsdT.N))

T = 0.1
mu = 0.1182817
L = 2*numpy.pi/numpy.sqrt(1.0)
norb = 7
cut = 1.2
damp = 0.1
mi = 50
ueg = UEGSCFSystem(T, L, cut, mu=mu, norb=norb, orbtype='u')
logging.info('Norb: {}'.format(len(ueg.basis.basis)))
logging.info('L: {:.10f}'.format(L))
logging.info('N0: {:.10f}'.format(ueg.N))
logging.info('mu: {:.10f}'.format(ueg.mu))
logging.info('density: {:.10f}'.format(ueg.den))
logging.info('r_s: {:.10f}'.format(ueg.rs))
logging.info('T_F: {:.10f}'.format(ueg.Tf))

ccsdT = ccsd(
    ueg, T=T, mu=mu, iprint=1, max_iter=mi, damp=damp, ngrid=ng, tconv=1e-8)
Ecctot, Ecc = ccsdT.run()
logging.info('Omega: {:.10f}'.format(Ecctot))
logging.info('OmegaC: {:.12f}'.format(Ecc))

ccsdT.compute_ESN()
logging.info("E: {}".format(ccsdT.E))
logging.info("S: {}".format(ccsdT.S))
logging.info("N: {}".format(ccsdT.N))
