from pyscf import gto, scf, cc
from kelvin.mp3 import mp3
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

T = 0.1
mu = -0.2
sys = scf_system(m,T,mu)
ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=45,damp=0.2,ngrid=100,econv=1e-9)
Ecctot,Ecc = ccsdT.run()
ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=45,damp=0.2,ngrid=100,quad='ln',econv=1e-9)
Ecctot,Ecc_ln = ccsdT.run()
ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=45,damp=0.2,ngrid=100,quad='sin',econv=1e-9)
Ecctot,Ecc_sin = ccsdT.run()
ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=45,damp=0.2,ngrid=100,quad='quad',econv=1e-9)
Ecctot,Ecc_quad = ccsdT.run()
print(Ecc)
print(Ecc_ln)
print(Ecc_sin)
print(Ecc_quad)
