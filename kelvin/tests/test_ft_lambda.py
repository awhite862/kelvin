import unittest
import numpy
from pyscf import gto, scf, cc
from cqcpy import spin_utils
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from kelvin import cc_utils
from kelvin import ft_cc_energy
from kelvin import ft_cc_equations
from kelvin import quadrature

def test_L1(cc,thresh):
    # test lambdas
    T = cc.T
    beta = 1.0 / (T + 1e-12)
    mu = cc.mu
    ng = cc.ngrid
    delta = beta/(ng - 1.0)
    G = quadrature.get_G(ng,delta)
    g = quadrature.get_gint(ng, delta)
    ti = numpy.asarray([float(i)*delta for i in range(ng)])
    en = cc.sys.g_energies_tot()
    D1 = en[:,None] - en[None,:]
    D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
    F,I = cc_utils.get_ft_integrals(cc.sys, en, beta, mu)
    n = cc.L2.shape[1]
    d = 1e-4
    for y in range(ng):
        for i in range(n):
            for a in range(n):
                TF = cc.T1.copy()
                TB = cc.T1.copy()
                TF[y,a,i] += d
                TB[y,a,i] -= d
                EF = ft_cc_energy.ft_cc_energy(TF,cc.T2,F.ov,I.oovv,g,beta)
                EB = ft_cc_energy.ft_cc_energy(TB,cc.T2,F.ov,I.oovv,g,beta)
                TF1,TF2 = ft_cc_equations.ccsd_stanton(F,I,TF,cc.T2,D1,D2,ti,ng,G)
                TB1,TB2 = ft_cc_equations.ccsd_stanton(F,I,TB,cc.T2,D1,D2,ti,ng,G)
                TF2 -= cc.T2
                TB2 -= cc.T2
                TF1 -= TF
                TB1 -= TB
                TEf = 0.25*numpy.einsum('yijab,yabij->y',cc.L2, TF2)
                TEb = 0.25*numpy.einsum('yijab,yabij->y',cc.L2, TB2)
                TEf += numpy.einsum('yia,yai->y',cc.L1,TF1)
                TEb += numpy.einsum('yia,yai->y',cc.L1,TB1)
                g = quadrature.get_gint(ng, delta)
                Tef = (1.0/beta)*numpy.einsum('y,y->',TEf,g)
                Teb = (1.0/beta)*numpy.einsum('y,y->',TEb,g)
                fw = EF + Tef
                bw = EB + Teb
                diff = (fw - bw)/(2*d)
                if numpy.abs(diff) > thresh:
                    return ('{} {} {}: {}'.format(y,i,a,diff),False)
    return ("pass",True)

def test_L2(cc,thresh):
    # test lambdas
    T = cc.T
    beta = 1.0 / (T + 1e-12)
    mu = cc.mu
    ng = cc.ngrid
    delta = beta/(ng - 1.0)
    G = quadrature.get_G(ng,delta)
    g = quadrature.get_gint(ng, delta)
    ti = numpy.asarray([float(i)*delta for i in range(ng)])
    en = cc.sys.g_energies_tot()
    D1 = en[:,None] - en[None,:]
    D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
    F,I = cc_utils.get_ft_integrals(cc.sys, en, beta, mu)
    n = cc.L2.shape[1]
    d = 1e-4
    for y in range(ng):
        for i in range(n):
            for j in range(n):
                for a in range(n):
                    for b in range(n):
                        TF = cc.T2.copy()
                        TB = cc.T2.copy()
                        TF[y,a,b,i,j] += d
                        TF[y,a,b,j,i] -= d
                        TF[y,b,a,i,j] -= d
                        TF[y,b,a,j,i] += d
                        TB[y,a,b,i,j] -= d
                        TB[y,a,b,j,i] += d
                        TB[y,b,a,i,j] += d
                        TB[y,b,a,j,i] -= d
                        EF = ft_cc_energy.ft_cc_energy(cc.T1,TF,F.ov,I.oovv,g,beta)
                        EB = ft_cc_energy.ft_cc_energy(cc.T1,TB,F.ov,I.oovv,g,beta)
                        TF1,TF2 = ft_cc_equations.ccsd_stanton(F,I,cc.T1,TF,D1,D2,ti,ng,G)
                        TB1,TB2 = ft_cc_equations.ccsd_stanton(F,I,cc.T1,TB,D1,D2,ti,ng,G)
                        TF2 -= TF
                        TB2 -= TB
                        TF1 -= cc.T1
                        TB1 -= cc.T1
                        TEf = 0.25*numpy.einsum('yijab,yabij->y',cc.L2, TF2)
                        TEb = 0.25*numpy.einsum('yijab,yabij->y',cc.L2, TB2)
                        TEf += numpy.einsum('yia,yai->y',cc.L1,TF1)
                        TEb += numpy.einsum('yia,yai->y',cc.L1,TB1)
                        g = quadrature.get_gint(ng, delta)
                        Tef = (1.0/beta)*numpy.einsum('y,y->',TEf,g)
                        Teb = (1.0/beta)*numpy.einsum('y,y->',TEb,g)
                        fw = EF + Tef
                        bw = EB + Teb
                        diff = (fw - bw)/(2*d)
                        if numpy.abs(diff) > 1e-7:
                            return ('{} {} {} {} {}: {}'.format(y,i,j,a,b,diff),False)
    return ("pass",True)

class FTLambdaTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8
        self.T = 2.0
        self.mu = 0.05

    def test_Be_sto3g_gen(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        sys = scf_system(m,self.T,self.mu,orbtype='g')
        ccsdT = ccsd(sys,T=self.T,mu=self.mu,iprint=0,max_iter=44,econv=1e-12)
        Etot,Ecc = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        out = test_L1(ccsdT, self.thresh)
        #out = test_L2(ccsdT, self.thresh)
        self.assertTrue(out[1],out[0]) 

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        ng = 10
        sys = scf_system(m,self.T,self.mu,orbtype='u')
        ccsdT = ccsd(sys,T=self.T,mu=self.mu,ngrid=ng,iprint=0,max_iter=44,econv=1e-12)
        Etot,Ecc = ccsdT.run()
        ccsdT._ft_uccsd_lambda()
        ea,eb = ccsdT.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]
        n = na + nb

        # convert to spin orbitals
        L1 = numpy.zeros((ng,n,n))
        L2 = numpy.zeros((ng,n,n,n,n))
        T1 = numpy.zeros((ng,n,n))
        T2 = numpy.zeros((ng,n,n,n,n))
        for y in range(ng):
            L1[y] = spin_utils.T1_to_spin(ccsdT.L1[0][y],ccsdT.L1[1][y],na,na,nb,nb)
            L2[y] = spin_utils.T2_to_spin(ccsdT.L2[0][y],ccsdT.L2[1][y],ccsdT.L2[2][y],na,na,nb,nb)
            T1[y] = spin_utils.T1_to_spin(ccsdT.T1[0][y],ccsdT.T1[1][y],na,na,nb,nb)
            T2[y] = spin_utils.T2_to_spin(ccsdT.T2[0][y],ccsdT.T2[1][y],ccsdT.T2[2][y],na,na,nb,nb)
        nsys = scf_system(m,self.T,self.mu,orbtype='g')
        nccsdT = ccsd(sys,T=self.T,mu=self.mu,iprint=0,max_iter=44,econv=1e-12)
        nccsdT.L1 = L1
        nccsdT.L2 = L2
        nccsdT.T1 = T1
        nccsdT.T2 = T2
        out = test_L1(nccsdT, self.thresh)
        #out = test_L2(nccsdT, self.thresh)
        self.assertTrue(out[1],out[0])


if __name__ == '__main__':
    unittest.main()
