import logging
import sys as csys
from pyscf import gto, scf
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

T = 0.1
mu = -0.2
sys = SCFSystem(m, T, mu)
ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mu, max_iter=45,
    damp=0.2, ngrid=100, econv=1e-9)
Ecctot, Ecc = ccsdT.run()
ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mu, max_iter=45,
    damp=0.2, ngrid=100, quad='ln', econv=1e-9)
Ecctot, Ecc_ln = ccsdT.run()
ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mu, max_iter=45,
    damp=0.2, ngrid=100, quad='sin', econv=1e-9)
Ecctot, Ecc_sin = ccsdT.run()
ccsdT = ccsd(
    sys, iprint=1, T=T, mu=mu, max_iter=45,
    damp=0.2, ngrid=100, quad='quad', econv=1e-9)
Ecctot, Ecc_quad = ccsdT.run()
logging.info("{}".format(Ecc))
logging.info("{}".format(Ecc_ln))
logging.info("{}".format(Ecc_sin))
logging.info("{}".format(Ecc_quad))
