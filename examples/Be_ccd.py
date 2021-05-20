import logging
import sys as csys
from pyscf import gto, scf
from kelvin.mp3 import MP3
from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    stream=csys.stdout)

mol = gto.M(
    verbose=0,
    atom='Be 0 0 0',
    basis='sto-3G')

m = scf.RHF(mol)
scf.conv_tol_grad = 1e-12
m.conv_tol = 1e-12
logging.info('SCF energy: %f' % m.scf())

sys = SCFSystem(m, 0.0, 0.0, orbtype='g')
ccsd0 = ccsd(sys, iprint=1, max_iter=14, econv=1e-10, singles=False)
ccsd0.run()

T = 5.0
mu = 0.0
sys = SCFSystem(m, T, mu, orbtype='g')
mp3T = MP3(sys, iprint=1, T=T, mu=mu)
E0T, E1T, E2T, E3T = mp3T.run()
logging.info('HF energy: %.8f' % (E0T + E1T))
logging.info('MP3 correlation energy: %.8f' % (E2T + E3T))

ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mu, max_iter=35, damp=0.0, ngrid=10, singles=False)
Ecctot, Ecc = ccsdT.run()

logging.info("{} {}".format(E2T, E2T+E3T))
logging.info("{}".format(Ecc))
