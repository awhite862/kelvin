import time
from pyscf import gto, scf
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

mol = gto.M(
    verbose = 0,
    atom = 'Be 0 0 0',
    basis = 'sto-3G')

m = scf.RHF(mol)
scf.conv_tol_grad = 1e-12
m.conv_tol = 1e-12
print('SCF energy: %f' % m.scf())

T = 0.7
mu = 0.0
ng = 10
sys = scf_system(m,T,mu,orbtype='g')
t1 = time.time()
ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,singles=True)
Ecctot,Ecc = ccsdT.run()
ccsdT.compute_ESN()
t2 = time.time()
print('N = {}'.format(ccsdT.N))
print('E = {}'.format(ccsdT.E))
print('S = {}'.format(ccsdT.S))
print("Total ttme: {} s".format(t2 - t1))
