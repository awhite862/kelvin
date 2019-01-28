import unittest
import numpy
from cqcpy import test_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy import integrals
from cqcpy import utils 
from cqcpy import spin_utils
from kelvin import quadrature
from kelvin import ft_cc_equations
from kelvin import ft_cc_energy
from kelvin import ft_cc_equations
from kelvin import system

def evalL(T1f,T1b,T1i,T2f,T2b,T2i,L1f,L1b,L1i,L2f,L2b,L2i,
        Ff,Fb,F,I,D1,D2,tir,tii,gr,gi,Gr,Gi,beta):
    ngr = gr.shape[0]
    ngi = gi.shape[0]
    E = ft_cc_energy.ft_cc_energy_neq(T1f,T1b,T1i,T2f,T2b,T2i,
            Ff.ov,Fb.ov,F.ov,I.oovv,gr,gi,beta)
    T1f_,T1b_,T1i_,T2f_,T2b_,T2i_ =\
        ft_cc_equations.neq_ccsd_simple(Ff,Fb,F,I,T1f,T1b,T1i,
                T2f,T2b,T2i,D1,D2,tir,tii,ngr,ngi,Gr,Gi)

    TEf = 0.25*numpy.einsum('yijab,yabij->y',L2f, T2f_)
    TEf += numpy.einsum('yia,yai->y',L1f,T1f_)
    TEb = 0.25*numpy.einsum('yijab,yabij->y',L2b, T2b_)
    TEb += numpy.einsum('yia,yai->y',L1b,T1b_)
    TEi = 0.25*numpy.einsum('yijab,yabij->y',L2i, T2i_)
    TEi += numpy.einsum('yia,yai->y',L1i,T1i_)

    Te = (1.j/beta)*numpy.einsum('y,y->',TEf,gr)
    Te -= (1.j/beta)*numpy.einsum('y,y->',TEb,gr)
    Te += (1.0/beta)*numpy.einsum('y,y->',TEi,gi)

    return E + Te

class NEQDensityTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8

    def test_den(self):

        ngr = 4 
        ngi = 4 
        n = 5
        beta = 1.0
        tmax = 0.1
        tf = 2
        assert(tf < ngr)
        T1f,T2f = test_utils.make_random_ft_T(ngr,n)
        T1b,T2b = test_utils.make_random_ft_T(ngr,n)
        T1i,T2i = test_utils.make_random_ft_T(ngi,n)
        L1f,L2f = test_utils.make_random_ft_T(ngr,n)
        L1b,L2b = test_utils.make_random_ft_T(ngr,n)
        L1i,L2i = test_utils.make_random_ft_T(ngi,n)
        T1f = T1f.astype(complex)
        T1b = T1b.astype(complex)
        T2f = T2f.astype(complex)
        T2b = T2b.astype(complex)
        L1f = L1f.astype(complex)
        L1b = L1b.astype(complex)
        L2f = L2f.astype(complex)
        L2b = L2b.astype(complex)
        D1,D2 = test_utils.make_random_ft_D(n)
        D1 = numpy.zeros((n,n))
        tii,gi,Gi = quadrature.midpoint(ngi, beta)
        tir,gr,Gr = quadrature.midpoint(ngr, tmax)
        Aov = numpy.random.random((ngr,n,n))
        Avv = numpy.random.random((ngr,n,n))
        Aoo = numpy.random.random((ngr,n,n))
        Avo = numpy.random.random((ngr,n,n))
        Aov = Aov.astype(complex)
        Avv = Avv.astype(complex)
        Aoo = Aoo.astype(complex)
        Avo = Avo.astype(complex)
        zzr = numpy.zeros((ngr,n,n),dtype=complex)
        zzi = numpy.zeros((n,n),dtype=complex)
        for i in range(ngr):
            if i != tf:
                Aov[i] = numpy.zeros((n,n))
                Avv[i] = numpy.zeros((n,n))
                Aoo[i] = numpy.zeros((n,n))
                Avo[i] = numpy.zeros((n,n))
        # for the Lagrangian, divide through by measure
        Ftemp = one_e_blocks(Aoo/gr[tf],Aov/gr[tf],Avo/gr[tf],Avv/gr[tf])
        Fzr = one_e_blocks(zzr,zzr,zzr,zzr)
        Fzi = one_e_blocks(zzi,zzi,zzi,zzi)
        Inull = numpy.zeros((n,n,n,n),dtype=complex)
        I = two_e_blocks(
            vvvv=Inull, vvvo=Inull, vovv=Inull,
            vvoo=Inull, vovo=Inull, oovv=Inull,
            vooo=Inull, ooov=Inull, oooo=Inull)

        ref = evalL(T1f,T1b,T1i,T2f,T2b,T2i,L1f,L1b,L1i,L2f,L2b,L2i,
            Ftemp,Fzr,Fzi,I,D1,D2,tir,tii,gr,gi,Gr,Gi,beta)

        pia,pba,pji,pai = ft_cc_equations.neq_1rdm(
                T1f,T1b,T1i,T2f,T2b,T2i,
                L1f,L1b,L1i,L2f,L2b,L2i,
                D1,D2,tir,tii,ngr,ngi,gr,gi,Gr,Gi)

        out1 = 1.j*numpy.einsum('ai,ia->',Avo[tf],pia[tf])
        out2 = 1.j*numpy.einsum('ab,ba->',Avv[tf],pba[tf])
        out3 = 1.j*numpy.einsum('ij,ji->',Aoo[tf],pji[tf])
        out4 = 1.j*numpy.einsum('ia,ai->',Aov[tf],pai[tf])
        out = out1 + out2 + out3 + out4

        diff = numpy.linalg.norm(out - ref)
        msg = "Error: {}".format(diff)
        self.assertTrue(diff < self.thresh,msg)

if __name__ == '__main__':
    unittest.main()
