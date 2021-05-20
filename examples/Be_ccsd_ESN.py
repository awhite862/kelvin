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

sys = SCFSystem(m, 0.0, 0.0)
ccsd0 = ccsd(sys, iprint=1, max_iter=14, econv=1e-10)
ccsd0.run()

T = 2.0
mu = 0.0
ng = 10
sys = SCFSystem(m, T, mu)
mp3T = MP3(sys, iprint=1, T=T, mu=mu)
E0T, E1T, E2T, E3T = mp3T.run()
logging.info('HF energy: %.8f' % (E0T + E1T))
logging.info('MP3 correlation energy: %.8f' % (E2T + E3T))

ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mu, max_iter=35,
    damp=0.0, ngrid=ng, econv=1e-10, singles=True)
Ecctot, Ecc = ccsdT.run()
ccsdT.compute_ESN()
logging.info('N = {}'.format(ccsdT.N))
logging.info('E = {}'.format(ccsdT.E))
logging.info('S = {}'.format(ccsdT.S))

delta = 5e-4
muf = mu + delta
mub = mu - delta
sys = SCFSystem(m, T, muf)
ccsdT = ccsd(
    sys, iprint=1, T=T, mu=muf, max_iter=35, damp=0.0, ngrid=ng, econv=1e-10)
Ef, Ecf = ccsdT.run()
sys = SCFSystem(m, T, mub)
ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mub, max_iter=35, damp=0.0, ngrid=ng, econv=1e-10)
Eb, Ecb = ccsdT.run()

Nccx = -(Ecf - Ecb)/(2*delta)
Nx = -(Ef - Eb)/(2*delta)
logging.info("{}  {}".format(Nx - Nccx, Nccx))

Tf = T + delta
Tb = T - delta
sys = SCFSystem(m, Tf, mu)
ccsdT = ccsd(
    sys, iprint=1, T=Tf, mu=mu, max_iter=35, damp=0.0, ngrid=ng, econv=1e-10)
Ef, Ecf = ccsdT.run()
sys = SCFSystem(m, Tb, mu)
ccsdT = ccsd(
    sys, iprint=1, T=Tb, mu=mu, max_iter=35, damp=0.0, ngrid=ng, econv=1e-10)
Eb, Ecb = ccsdT.run()

Sccx = -(Ecf - Ecb)/(2*delta)
Sx = -(Ef - Eb)/(2*delta)
logging.info("{}  {}".format(Sx, Sccx))
Eccx = Ecc + T*Sccx + mu*Nccx
Ex = Ecctot + T*Sx + mu*Nx

logging.info('N = {}'.format(Nx))
logging.info('E = {}'.format(Ex))
logging.info('S = {}'.format(Sx))
