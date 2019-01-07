import unittest
import numpy
from cqcpy import test_utils
from kelvin import ft_cc_equations
from kelvin import ft_cc_energy
from kelvin import quadrature

einsum = numpy.einsum

def compute_ref(T1,T2,L1,L2,F,I,D1,D2,ti,ng,g,G,beta):
    T1temp,T2temp = ft_cc_equations.ccsd_simple(F,I,T1,T2,
            D1,D2,ti,ng,G)

    Eterm = ft_cc_energy.ft_cc_energy(T1,T2,F.ov,I.oovv,ti,g,beta)
    A1 = (1.0/beta)*einsum('via,vai->v',L1, T1temp)
    A2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2, T2temp)
    A1g = einsum('v,v->',A1,g)
    A2g = einsum('v,v->',A2,g)
    return A1g + A2g + Eterm

class FTCCSD_RDMTest(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.thresh = 1e-10
        self.ng = 10
        self.beta = 2.0

    def test_dia(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        F.oo = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)
        out = (1.0/beta)*numpy.einsum('ia,ai->',pia,F.vo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pia: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dba(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)
        out = (1.0/beta)*numpy.einsum('ba,ab->',pba,F.vv)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pia: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dji(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.ov = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)
        out = (1.0/beta)*numpy.einsum('ji,ij->',pji,F.oo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pia: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dai(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)
        out = (1.0/beta)*numpy.einsum('ai,ia->',pai,F.ov)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pia: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dcdab(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        #I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pcdab = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[0]
        out = (0.25/beta)*numpy.einsum('cdab,abcd->',Pcdab,I.vvvv)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pcdab: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dciab(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        #I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pciab = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[1]
        out = (0.5/beta)*numpy.einsum('ciab,abci->',Pciab,I.vvvo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pciab: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dbcai(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        #I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pbcai = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[2]
        out = (0.5/beta)*numpy.einsum('bcai,aibc->',Pbcai,I.vovv)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pbcai: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dijab(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        #I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pijab = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[3]
        out = (0.25/beta)*numpy.einsum('ijab,abij->',Pijab,I.vvoo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pijab: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dbjai(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        #I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pbjai = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[4]
        out = (1.0/beta)*numpy.einsum('bjai,aibj->',Pbjai,I.vovo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pbjai: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dabij(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        #I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pabij = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[5]
        out = (0.25/beta)*numpy.einsum('abij,ijab->',Pabij,I.oovv)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pabij: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_djkai(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        #I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pjkai = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[6]
        out = (0.5/beta)*numpy.einsum('jkai,aijk->',Pjkai,I.vooo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pjkai: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dkaij(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        #I.ooov = numpy.zeros((n,n,n,n))
        I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pkaij = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[7]
        out = (0.5/beta)*numpy.einsum('kaij,ijka->',Pkaij,I.ooov)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pkaij: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_dklij(self):
        n = self.n
        ng = self.ng
        beta = self.beta
        T1old,T2old = test_utils.make_random_ft_T(ng,n)
        L1old,L2old = test_utils.make_random_ft_T(ng,n)
        F,I = test_utils.make_random_integrals(n,n)
        F.oo = numpy.zeros((n,n))
        F.vo = numpy.zeros((n,n))
        F.ov = numpy.zeros((n,n))
        F.vv = numpy.zeros((n,n))
        I.vvvv = numpy.zeros((n,n,n,n))
        I.vvvo = numpy.zeros((n,n,n,n))
        I.vovv = numpy.zeros((n,n,n,n))
        I.vvoo = numpy.zeros((n,n,n,n))
        I.vovo = numpy.zeros((n,n,n,n))
        I.oovv = numpy.zeros((n,n,n,n))
        I.vooo = numpy.zeros((n,n,n,n))
        I.ooov = numpy.zeros((n,n,n,n))
        #I.oooo = numpy.zeros((n,n,n,n))

        D1,D2 = test_utils.make_random_ft_D(n)
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng, delta)
        g = quadrature.get_gint(ng, delta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms 
        Pklij = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,delta)[8]
        out = (0.25/beta)*numpy.einsum('klij,ijkl->',Pklij,I.oooo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pklij: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

if __name__ == '__main__':
    unittest.main()
