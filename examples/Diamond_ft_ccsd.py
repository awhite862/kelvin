from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from pyscf.pbc import gto, scf

T = 0.1
mu = 0.11
cell = gto.Cell()
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

mf = scf.RHF(cell, exxdiv=None)
mf.conv_tol_grad = 1e-8
mf.conv_tol = 1e-12
Escf = mf.kernel()
sys = scf_system(mf, T, mu)
ccsdT = ccsd(sys, iprint=1, max_iter=100, econv=1e-11, damp=0.0, T=T, mu=mu)
Etot, Ecc = ccsdT.run()
ccsdT.compute_ESN()
print('N = {}'.format(ccsdT.N))
print('E = {}'.format(ccsdT.E))
print('S = {}'.format(ccsdT.S))
