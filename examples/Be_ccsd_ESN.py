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

sys = scf_system(m,0.0,0.0)
ccsd0 = ccsd(sys,iprint=1,max_iter=14,econv=1e-10)
ccsd0.run()

T = 2.0
mu = 0.0
ng = 10
sys = scf_system(m,T,mu)
mp3T = mp3(sys,iprint=1,T=T,mu=mu)
E0T,E1T,E2T,E3T = mp3T.run()
print('HF energy: %.8f' % (E0T + E1T))
print('MP3 correlation energy: %.8f' % (E2T + E3T))

ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,singles=True)
Ecctot,Ecc = ccsdT.run()
ccsdT.compute_ESN()
print('N = {}'.format(ccsdT.N))
print('E = {}'.format(ccsdT.E))
print('S = {}'.format(ccsdT.S))

#print(E2T,E2T+E3T)
#print(Ecc)

delta = 5e-4
muf = mu + delta
mub = mu - delta
sys = scf_system(m,T,muf)
ccsdT = ccsd(sys,iprint=1,T=T,mu=muf,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
Ef,Ecf = ccsdT.run()
sys = scf_system(m,T,mub)
ccsdT = ccsd(sys,iprint=1,T=T,mu=mub,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
Eb,Ecb = ccsdT.run()

Nccx = -(Ecf - Ecb)/(2*delta)
Nx = -(Ef - Eb)/(2*delta)
#N1x = Nx - N0 - Nccx

Tf = T + delta
Tb = T - delta
sys = scf_system(m,Tf,mu)
ccsdT = ccsd(sys,iprint=1,T=Tf,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
Ef,Ecf = ccsdT.run()
sys = scf_system(m,Tb,mu)
ccsdT = ccsd(sys,iprint=1,T=Tb,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
Eb,Ecb = ccsdT.run()

Sccx = -(Ecf - Ecb)/(2*delta)
Sx = -(Ef - Eb)/(2*delta)
print(Sx,Sccx)
Eccx = Ecc + T*Sccx + mu*Nccx
Ex = Ecctot + T*Sx + mu*Nx

print('N = {}'.format(Nx))
print('E = {}'.format(Ex))
print('S = {}'.format(Sx))
