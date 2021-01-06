import unittest
import numpy
from cqcpy import test_utils
from cqcpy import spin_utils
from kelvin import ft_cc_equations
from kelvin import ft_cc_energy
from kelvin import quadrature

einsum = numpy.einsum

def compute_ref(T1,T2,L1,L2,F,I,D1,D2,ti,ng,g,G,beta):
    T1temp,T2temp = ft_cc_equations.ccsd_simple(F,I,T1,T2,
            D1,D2,ti,ng,G)

    Eterm = ft_cc_energy.ft_cc_energy(T1,T2,F.ov,I.oovv,g,beta)
    A1 = (1.0/beta)*einsum('via,vai->v',L1, T1temp)
    A2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2, T2temp)
    A1g = einsum('v,v->',A1,g)
    A2g = einsum('v,v->',A2,g)
    return Eterm - A1g - A2g

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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)
        out = (1.0/beta)*numpy.einsum('ba,ab->',pba,F.vv)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pba: {}".format(diff)
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)
        out = (1.0/beta)*numpy.einsum('ji,ij->',pji,F.oo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pji: {}".format(diff)
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)
        out = (1.0/beta)*numpy.einsum('ai,ia->',pai,F.ov)

        diff = abs(out - ref)/abs(ref)
        error = "Error in pai: {}".format(diff)
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pcdab = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[0]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pciab = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[1]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pbcai = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[2]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pijab = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[3]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pbjai = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[4]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pabij = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[5]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pjkai = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[6]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pkaij = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[7]
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
        ti,g,G = quadrature.simpsons(ng, beta)

        # compute the trace from the CC equations
        ref = compute_ref(T1old,T2old,L1old,L2old,F,I,D1,D2,ti,ng,g,G,beta)
        # compute the trace from the rdms
        Pklij = ft_cc_equations.ccsd_2rdm(T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G)[8]
        out = (0.25/beta)*numpy.einsum('klij,ijkl->',Pklij,I.oooo)

        diff = abs(out - ref)/abs(ref)
        error = "Error in Pklij: {}".format(diff)
        self.assertTrue(diff < self.thresh, error)

    def test_u1rdm(self):
        ng = 4
        na = 8
        nb = 8
        thresh = 1e-14
        beta = 2.0
        ti,g,G = quadrature.simpsons(ng, beta)
        n = na + nb
        T1a,T1b = test_utils.make_random_ft_T1_spatial(ng,na,nb)
        L1a,L1b = test_utils.make_random_ft_T1_spatial(ng,na,nb)
        T2aa,T2ab,T2bb = test_utils.make_random_ft_T2_spatial(ng,na,nb)
        L2aa,L2ab,L2bb = test_utils.make_random_ft_T2_spatial(ng,na,nb)
        D1a = test_utils.make_random_ft_D1(na)
        D1b = test_utils.make_random_ft_D1(nb)
        D2aa = test_utils.make_random_ft_D2(na,na)
        D2ab = test_utils.make_random_ft_D2(na,nb)
        D2bb = test_utils.make_random_ft_D2(nb,nb)
        T1 = numpy.zeros((ng,n,n))
        L1 = numpy.zeros((ng,n,n))
        T2 = numpy.zeros((ng,n,n,n,n))
        L2 = numpy.zeros((ng,n,n,n,n))
        D1 = spin_utils.T1_to_spin(D1a, D1b, na, na, nb, nb)
        D2 = spin_utils.D2_to_spin(D2aa, D2ab, D2bb, na, na, nb, nb)
        for i in range(ng):
            T1[i] = spin_utils.T1_to_spin(T1a[i], T1b[i], na, na, nb, nb)
            L1[i] = spin_utils.T1_to_spin(L1a[i], L1b[i], na, na, nb, nb)
            T2[i] = spin_utils.T2_to_spin(T2aa[i], T2ab[i], T2bb[i], na, na, nb, nb)
            L2[i] = spin_utils.T2_to_spin(L2aa[i], L2ab[i], L2bb[i], na, na, nb, nb)

        urdm1 = ft_cc_equations.uccsd_1rdm(T1a,T1b,T2aa,T2ab,T2bb,
                L1a,L1b,L2aa,L2ab,L2bb,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,g,G)
        grdm1 = ft_cc_equations.ccsd_1rdm(T1,T2,L1,L2,D1,D2,ti,ng,g,G)

        # test ia
        refa = grdm1[0][:na,:na]
        refb = grdm1[0][na:,na:]
        diffa = numpy.linalg.norm(refa - urdm1[0][0])/numpy.linalg.norm(refa)
        diffb = numpy.linalg.norm(refb - urdm1[0][1])/numpy.linalg.norm(refa)
        self.assertTrue(diffa < thresh)
        self.assertTrue(diffb < thresh)
        # test ba
        refa = grdm1[1][:na,:na]
        refb = grdm1[1][na:,na:]
        diffa = numpy.linalg.norm(refa - urdm1[1][0])/numpy.linalg.norm(refa)
        diffb = numpy.linalg.norm(refb - urdm1[1][1])/numpy.linalg.norm(refa)
        self.assertTrue(diffa < thresh)
        self.assertTrue(diffb < thresh)
        # test ji
        refa = grdm1[2][:na,:na]
        refb = grdm1[2][na:,na:]
        diffa = numpy.linalg.norm(refa - urdm1[2][0])/numpy.linalg.norm(refa)
        diffb = numpy.linalg.norm(refb - urdm1[2][1])/numpy.linalg.norm(refa)
        self.assertTrue(diffa < thresh)
        self.assertTrue(diffb < thresh)
        # test ai
        refa = grdm1[3][:na,:na]
        refb = grdm1[3][na:,na:]
        diffa = numpy.linalg.norm(refa - urdm1[3][0])/numpy.linalg.norm(refa)
        diffb = numpy.linalg.norm(refb - urdm1[3][1])/numpy.linalg.norm(refa)
        self.assertTrue(diffa < thresh)
        self.assertTrue(diffb < thresh)

    def test_u2rdm(self):
        ng = 4
        na = 8
        nb = 8
        thresh = 1e-14
        beta = 2.0
        ti,g,G = quadrature.simpsons(ng, beta)
        n = na + nb
        T1a,T1b = test_utils.make_random_ft_T1_spatial(ng,na,nb)
        L1a,L1b = test_utils.make_random_ft_T1_spatial(ng,na,nb)
        T2aa,T2ab,T2bb = test_utils.make_random_ft_T2_spatial(ng,na,nb)
        L2aa,L2ab,L2bb = test_utils.make_random_ft_T2_spatial(ng,na,nb)
        D1a = test_utils.make_random_ft_D1(na)
        D1b = test_utils.make_random_ft_D1(nb)
        D2aa = test_utils.make_random_ft_D2(na,na)
        D2ab = test_utils.make_random_ft_D2(na,nb)
        D2bb = test_utils.make_random_ft_D2(nb,nb)
        T1 = numpy.zeros((ng,n,n))
        L1 = numpy.zeros((ng,n,n))
        T2 = numpy.zeros((ng,n,n,n,n))
        L2 = numpy.zeros((ng,n,n,n,n))
        D1 = spin_utils.T1_to_spin(D1a, D1b, na, na, nb, nb)
        D2 = spin_utils.D2_to_spin(D2aa, D2ab, D2bb, na, na, nb, nb)
        for i in range(ng):
            T1[i] = spin_utils.T1_to_spin(T1a[i], T1b[i], na, na, nb, nb)
            L1[i] = spin_utils.T1_to_spin(L1a[i], L1b[i], na, na, nb, nb)
            T2[i] = spin_utils.T2_to_spin(T2aa[i], T2ab[i], T2bb[i], na, na, nb, nb)
            L2[i] = spin_utils.T2_to_spin(L2aa[i], L2ab[i], L2bb[i], na, na, nb, nb)

        urdm2 = ft_cc_equations.uccsd_2rdm(T1a,T1b,T2aa,T2ab,T2bb,
                L1a,L1b,L2aa,L2ab,L2bb,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,g,G)
        grdm2 = ft_cc_equations.ccsd_2rdm(T1,T2,L1,L2,D1,D2,ti,ng,g,G)

        # test cdab
        ref1 = grdm2[0][:na,:na,:na,:na]
        ref2 = grdm2[0][na:,na:,na:,na:]
        ref3 = grdm2[0][:na,na:,:na,na:]
        out1 = urdm2[0][0]
        out2 = urdm2[0][1]
        out3 = urdm2[0][2]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        self.assertTrue(diff1 < thresh, "error in Pcdab: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PCDAB: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PcDaB: {}".format(diff3))

        # test ciab
        ref1 = grdm2[1][:na,:na,:na,:na]
        ref2 = grdm2[1][na:,na:,na:,na:]
        ref3 = grdm2[1][:na,na:,:na,na:]
        ref4 = grdm2[1][na:,:na,na:,:na]
        out1 = urdm2[1][0]
        out2 = urdm2[1][1]
        out3 = urdm2[1][2]
        out4 = urdm2[1][3]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        diff4 = numpy.linalg.norm(ref4 - out4)/numpy.linalg.norm(ref4)
        self.assertTrue(diff1 < thresh, "error in Pciab: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PCIAB: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PcIaB: {}".format(diff3))
        self.assertTrue(diff4 < thresh, "error in PCiAb: {}".format(diff4))

        # test bcai
        ref1 = grdm2[2][:na,:na,:na,:na]
        ref2 = grdm2[2][na:,na:,na:,na:]
        ref3 = grdm2[2][:na,na:,:na,na:]
        ref4 = grdm2[2][na:,:na,na:,:na]
        out1 = urdm2[2][0]
        out2 = urdm2[2][1]
        out3 = urdm2[2][2]
        out4 = urdm2[2][3]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        diff4 = numpy.linalg.norm(ref4 - out4)/numpy.linalg.norm(ref4)
        self.assertTrue(diff1 < thresh, "error in Pbcai: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PBCAI: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PbCaI: {}".format(diff3))
        self.assertTrue(diff4 < thresh, "error in PBcAi: {}".format(diff4))

        # test ijab
        ref1 = grdm2[3][:na,:na,:na,:na]
        ref2 = grdm2[3][na:,na:,na:,na:]
        ref3 = grdm2[3][:na,na:,:na,na:]
        out1 = urdm2[3][0]
        out2 = urdm2[3][1]
        out3 = urdm2[3][2]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        self.assertTrue(diff1 < thresh, "error in Pijab: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PIJAB: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PiJaB: {}".format(diff3))

        # test bjai
        ref1 = grdm2[4][:na,:na,:na,:na]
        ref2 = grdm2[4][na:,na:,na:,na:]
        ref3 = grdm2[4][:na,na:,:na,na:]
        ref4 = grdm2[4][:na,na:,na:,:na]
        ref5 = grdm2[4][na:,:na,:na,na:]
        ref6 = grdm2[4][na:,:na,na:,:na]
        out1 = urdm2[4][0]
        out2 = urdm2[4][1]
        out3 = urdm2[4][2]
        out4 = urdm2[4][3]
        out5 = urdm2[4][4]
        out6 = urdm2[4][5]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        diff4 = numpy.linalg.norm(ref4 - out4)/numpy.linalg.norm(ref4)
        diff5 = numpy.linalg.norm(ref5 - out5)/numpy.linalg.norm(ref5)
        diff6 = numpy.linalg.norm(ref6 - out6)/numpy.linalg.norm(ref6)
        self.assertTrue(diff1 < thresh, "error in Pbjai: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PBJAI: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PbJaI: {}".format(diff3))
        self.assertTrue(diff4 < thresh, "error in PbJAi: {}".format(diff4))
        self.assertTrue(diff5 < thresh, "error in PBjaI: {}".format(diff5))
        self.assertTrue(diff6 < thresh, "error in PBjAi: {}".format(diff6))

        # test abij
        ref1 = grdm2[5][:na,:na,:na,:na]
        ref2 = grdm2[5][na:,na:,na:,na:]
        ref3 = grdm2[5][:na,na:,:na,na:]
        out1 = urdm2[5][0]
        out2 = urdm2[5][1]
        out3 = urdm2[5][2]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        self.assertTrue(diff1 < thresh, "error in Pabij: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PABIJ: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PaBiJ: {}".format(diff3))

        # test jkai
        ref1 = grdm2[6][:na,:na,:na,:na]
        ref2 = grdm2[6][na:,na:,na:,na:]
        ref3 = grdm2[6][:na,na:,:na,na:]
        ref4 = grdm2[6][na:,:na,na:,:na]
        out1 = urdm2[6][0]
        out2 = urdm2[6][1]
        out3 = urdm2[6][2]
        out4 = urdm2[6][3]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        diff4 = numpy.linalg.norm(ref4 - out4)/numpy.linalg.norm(ref4)
        self.assertTrue(diff1 < thresh, "error in Pjkai: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PJKAI: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PjKaI: {}".format(diff3))
        self.assertTrue(diff4 < thresh, "error in PJkAi: {}".format(diff4))

        # test kaij
        ref1 = grdm2[7][:na,:na,:na,:na]
        ref2 = grdm2[7][na:,na:,na:,na:]
        ref3 = grdm2[7][:na,na:,:na,na:]
        ref4 = grdm2[7][na:,:na,na:,:na]
        out1 = urdm2[7][0]
        out2 = urdm2[7][1]
        out3 = urdm2[7][2]
        out4 = urdm2[7][3]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        diff4 = numpy.linalg.norm(ref4 - out4)/numpy.linalg.norm(ref4)
        self.assertTrue(diff1 < thresh, "error in Pkaij: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PKAIJ: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PkAiJ: {}".format(diff3))
        self.assertTrue(diff4 < thresh, "error in PKaIj: {}".format(diff4))

        # test klij
        ref1 = grdm2[8][:na,:na,:na,:na]
        ref2 = grdm2[8][na:,na:,na:,na:]
        ref3 = grdm2[8][:na,na:,:na,na:]
        out1 = urdm2[8][0]
        out2 = urdm2[8][1]
        out3 = urdm2[8][2]
        diff1 = numpy.linalg.norm(ref1 - out1)/numpy.linalg.norm(ref1)
        diff2 = numpy.linalg.norm(ref2 - out2)/numpy.linalg.norm(ref2)
        diff3 = numpy.linalg.norm(ref3 - out3)/numpy.linalg.norm(ref3)
        self.assertTrue(diff1 < thresh, "error in Pklij: {}".format(diff1))
        self.assertTrue(diff2 < thresh, "error in PKLIJ: {}".format(diff2))
        self.assertTrue(diff3 < thresh, "error in PkLiJ: {}".format(diff3))

if __name__ == '__main__':
    unittest.main()
