import unittest
import numpy
from pyscf import gto, scf
from cqcpy import ft_utils 
from cqcpy import integrals
from kelvin.h2_field_system import h2_field_system
from kelvin.neq_ccsd import neq_ccsd
from kelvin.ccsd import ccsd
from kelvin.h2_pol_system import h2_pol_system

class NEQ_CCSDTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 3e-4

    def test_h2_field(self):
        beta = 1.0
        T = 1./beta
        mu = 0.0
        omega = 0.5

        ngrid_ref = 4000
        deltat = 0.00025
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 -0.6; H 0 0 0.0',
            basis = 'STO-3G',
            charge = 1,
            spin = 1)
        
        m = scf.UHF(mol)
        Escf = m.scf()
        mos = m.mo_coeff[0]
        
        eri = integrals.get_phys(mol, mos, mos, mos, mos)
        hcore = numpy.einsum('mp,mn,nq->pq',mos,m.get_hcore(m.mol),mos)
        
        E = numpy.zeros((3))
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq',mos,field,mos)
        nuc_e = m.mol.energy_nuc() 
        H = numpy.zeros((4,4))
        H[1,1] += hcore[0,0]
        H[2,2] += hcore[1,1]
        H[1,2] += hcore[0,1]
        H[2,1] += hcore[1,0]
        H[3,3] = hcore[0,0] + hcore[1,1] + eri[0,1,0,1] - eri[0,1,1,0]
        Hint = numpy.zeros((4,4))
        Hint[1,1] = field[0,0]
        Hint[2,2] = field[1,1] 
        Hint[1,2] = field[0,1] 
        Hint[2,1] = field[1,0] 
        Hint[3,3] = field[0,0] + field[1,1]

        e0,v0 = numpy.linalg.eigh(H)
        exp = numpy.exp(-beta*(e0))
        Z = exp.sum()
        p0 = numpy.einsum('mi,i,ni->mn',v0,exp,v0) / Z
        p = p0.copy()
        ti = numpy.zeros(ngrid_ref)
        A_ref = []
        
        for i in range(ngrid_ref):
            t = i*deltat
            ti[i] = t
            Ht = H + numpy.sin(omega*t)*Hint
            e,v = numpy.linalg.eigh(Ht)
            ee = numpy.exp(-deltat*1.j*e)
            U = numpy.einsum('ai,i,bi->ab',v,ee,numpy.conj(v))
            p = numpy.einsum('ps,pq,qr->sr',numpy.conj(U),p,U)
            if i%(ngrid_ref//10) == 0:
                A_ref.append(numpy.einsum('ij,ji->',p,Hint))
                #print(t,A_ref[len(A_ref) - 1])

        del A_ref[0]
        # Neq-CCSD
        ng = 640
        ngi = 80
        A = []
        for i in range(9):
            tmax = (i + 1)*0.1
            deltat = tmax / (ng)
            d = 1e-4
            ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])
            sys = h2_field_system(T,mu,omega,ti,O=(d*field),ot=ng - 1)
            cc = neq_ccsd(sys,T,mu=mu,tmax=tmax,econv=1e-10,max_iter=40,damp=0.0,ngr=ng,ngi=ngi,iprint=0)
            Ef = cc.run()
        
            sys = h2_field_system(T,mu,omega,ti,O=(-d*field),ot=ng - 1)
            cc = neq_ccsd(sys,T,mu=mu,tmax=tmax,econv=1e-10,max_iter=40,damp=0.0,ngr=ng,ngi=ngi,iprint=0)
            Eb = cc.run()
            #print((Ef[0] + Eb[0])/2)
            A.append((Ef[0] - Eb[0])/(2*d))

        for i,out in enumerate(A):
            ref = A_ref[i]
            diff = abs(ref - out)
            #print("{} -- Expected: {}  Actual: {} ".format(i,ref,out))
            msg = "{} -- Expected: {}  Actual: {} ".format(i,ref,out)
            self.assertTrue(diff < self.thresh,msg)
        

if __name__ == '__main__':
    unittest.main()
