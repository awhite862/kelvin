import unittest
import numpy
from cqcpy.ov_blocks import one_e_blocks
from cqcpy import test_utils
from kelvin import ft_cc_energy
from kelvin import ft_cc_equations
from kelvin import quadrature


def evalL(T1f, T1b, T1i, T2f, T2b, T2i,
          L1f, L1b, L1i, L2f, L2b, L2i,
          Ff, Fb, F, I, D1, D2, tir, tii,
          gr, gi, Gr, Gi, beta, iprint=False):
    ngr = gr.shape[0]
    ngi = gi.shape[0]
    E = ft_cc_energy.ft_cc_energy_neq(
            T1f, T1b, T1i, T2f, T2b, T2i,
            Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
    T1f_, T1b_, T1i_, T2f_, T2b_, T2i_ =\
        ft_cc_equations.neq_ccsd_stanton(
            Ff, Fb, F, I, T1f, T1b, T1i,
            T2f, T2b, T2i, D1, D2, tir, tii, ngr, ngi, Gr, Gi)
    T1f_ -= T1f
    T1b_ -= T1b
    T1i_ -= T1i
    T2f_ -= T2f
    T2b_ -= T2b
    T2i_ -= T2i

    TEf = 0.25*numpy.einsum('yijab,yabij->y', L2f, T2f_)
    TEf += numpy.einsum('yia,yai->y', L1f, T1f_)
    TEb = 0.25*numpy.einsum('yijab,yabij->y', L2b, T2b_)
    TEb += numpy.einsum('yia,yai->y', L1b, T1b_)
    TEi = 0.25*numpy.einsum('yijab,yabij->y', L2i, T2i_)
    TEi += numpy.einsum('yia,yai->y', L1i, T1i_)

    Tef = (1.j/beta)*numpy.einsum('y,y->', TEf, gr)
    Teb = -(1.j/beta)*numpy.einsum('y,y->', TEb, gr)
    Tei = (1.0/beta)*numpy.einsum('y,y->', TEi, gi)
    Te = Tef + Teb + Tei
    return E + Te


class NEQLambdaEquationsTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8

    def test_L1_simple(self):
        ngr = 4
        ngi = 4
        n = 3
        beta = 2.0
        tmax = 0.1
        tii, gi, Gi = quadrature.simpsons(ngi, beta)
        tir, gr, Gr = quadrature.simpsons(ngr, tmax)
        T1f, T2f = test_utils.make_random_ft_T(ngr, n)
        T1b, T2b = test_utils.make_random_ft_T(ngr, n)
        T1i, T2i = test_utils.make_random_ft_T(ngi, n)
        L1f, L2f = test_utils.make_random_ft_T(ngr, n)
        L1b, L2b = test_utils.make_random_ft_T(ngr, n)
        L1i, L2i = test_utils.make_random_ft_T(ngi, n)
        T1f = T1f.astype(complex)
        T1b = T1b.astype(complex)
        T2f = T2f.astype(complex)
        T2b = T2b.astype(complex)
        L1f = L1f.astype(complex)
        L1b = L1b.astype(complex)
        L2f = L2f.astype(complex)
        L2b = L2b.astype(complex)
        F, I = test_utils.make_random_integrals(n, n)
        Foo = F.oo.astype(complex)
        Fov = F.ov.astype(complex)
        Fvo = F.vo.astype(complex)
        Fvv = F.vv.astype(complex)
        Ff = one_e_blocks(
                numpy.ones(ngr)[:, None, None]*Foo[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fov[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fvo[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fvv[None, :, :])
        I.vvvv = I.vvvv.astype(complex)
        I.vovo = I.vovo.astype(complex)
        I.vvoo = I.vvoo.astype(complex)
        I.oovv = I.oovv.astype(complex)
        I.oooo = I.oooo.astype(complex)
        F = one_e_blocks(Foo, Fov, Fvo, Fvv)
        Fb = Ff
        D1, D2 = test_utils.make_random_ft_D(n)

        # compute dL/dt1 from Lagrangian directly
        dT1i = numpy.zeros((ngi, n, n), dtype=complex)
        dT1b = numpy.zeros((ngr, n, n), dtype=complex)
        dT1f = numpy.zeros((ngr, n, n), dtype=complex)
        for y in range(ngi):
            for i in range(n):
                for a in range(n):
                    d = 1.e-4
                    TP = T1i.copy()
                    TM = T1i.copy()
                    TP[y, a, i] += d
                    TM[y, a, i] -= d
                    LP = evalL(
                        T1f, T1b, TP, T2f, T2b, T2i,
                        L1f, L1b, L1i, L2f, L2b, L2i,
                        Ff, Fb, F, I, D1, D2, tir, tii,
                        gr, gi, Gr, Gi, beta)
                    LM = evalL(
                        T1f, T1b, TM, T2f, T2b, T2i,
                        L1f, L1b, L1i, L2f, L2b, L2i,
                        Ff, Fb, F, I, D1, D2, tir, tii,
                        gr, gi, Gr, Gi, beta)
                    dT1i[y, i, a] = (LP - LM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for a in range(n):
                    d = 1.e-4
                    TP = T1b.copy()
                    TM = T1b.copy()
                    TP[y, a, i] += d
                    TM[y, a, i] -= d
                    LP = evalL(
                        T1f, TP, T1i, T2f, T2b, T2i,
                        L1f, L1b, L1i, L2f, L2b, L2i,
                        Ff, Fb, F, I, D1, D2, tir, tii,
                        gr, gi, Gr, Gi, beta)
                    LM = evalL(
                        T1f, TM, T1i, T2f, T2b, T2i,
                        L1f, L1b, L1i, L2f, L2b, L2i,
                        Ff, Fb, F, I, D1, D2, tir, tii,
                        gr, gi, Gr, Gi, beta)
                    dT1b[y, i, a] = (LP - LM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for a in range(n):
                    TP = T1f.copy()
                    TM = T1f.copy()
                    TP[y, a, i] += d
                    TM[y, a, i] -= d
                    LP = evalL(
                        TP, T1b, T1i, T2f, T2b, T2i,
                        L1f, L1b, L1i, L2f, L2b, L2i,
                        Ff, Fb, F, I, D1, D2, tir, tii,
                        gr, gi, Gr, Gi, beta)
                    LM = evalL(
                        TM, T1b, T1i, T2f, T2b, T2i,
                        L1f, L1b, L1i, L2f, L2b, L2i,
                        Ff, Fb, F, I, D1, D2, tir, tii,
                        gr, gi, Gr, Gi, beta)
                    dT1f[y, i, a] = (LP - LM)/(2*d)

        # compute dL/dt1 from the Lambda equations
        L1fn, L1bn, L1in, L2fn, L2bn, L2in = \
            ft_cc_equations.neq_lambda_simple(
                Ff, Fb, F, I, L1f, L1b, L1i, L2f, L2b, L2i, T1f, T1b, T1i,
                T2f, T2b, T2i, D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi)

        L1in -= L1i
        L1bn -= L1b
        L1fn -= L1f
        for y in range(ngi):
            L1in[y] *= gi[y]/beta
        for y in range(ngr):
            L1bn[y] *= -1.j*gr[y]/beta
            L1fn[y] *= 1.j*gr[y]/beta

        diffi = numpy.linalg.norm(L1in - dT1i)
        diffb = numpy.linalg.norm(L1bn - dT1b)
        difff = numpy.linalg.norm(L1fn - dT1f)
        self.assertTrue(
            abs(diffi) < self.thresh, "Error in dTi: {}".format(diffi))
        self.assertTrue(
            abs(diffb) < self.thresh, "Error in dTb: {}".format(diffb))
        self.assertTrue(
            abs(difff) < self.thresh, "Error in dTf: {}".format(difff))

    def test_L2_simple(self):
        ngr = 4
        ngi = 4
        n = 2
        beta = 2.0
        tmax = 0.1
        T1f, T2f = test_utils.make_random_ft_T(ngr, n)
        T1b, T2b = test_utils.make_random_ft_T(ngr, n)
        T1i, T2i = test_utils.make_random_ft_T(ngi, n)
        L1f, L2f = test_utils.make_random_ft_T(ngr, n)
        L1b, L2b = test_utils.make_random_ft_T(ngr, n)
        L1i, L2i = test_utils.make_random_ft_T(ngi, n)
        T1f = T1f.astype(complex)
        T1b = T1b.astype(complex)
        T2f = T2f.astype(complex)
        T2b = T2b.astype(complex)
        L1f = L1f.astype(complex)
        L1b = L1b.astype(complex)
        L2f = L2f.astype(complex)
        L2b = L2b.astype(complex)
        F, I = test_utils.make_random_integrals(n, n)
        I.vvvv = I.vvvv.astype(complex)
        I.oovv = I.oovv.astype(complex)
        I.vvoo = I.vvoo.astype(complex)
        I.vovo = I.vovo.astype(complex)
        I.oooo = I.oooo.astype(complex)
        Foo = F.oo.astype(complex)
        Fov = F.ov.astype(complex)
        Fvo = F.vo.astype(complex)
        Fvv = F.vv.astype(complex)
        Ff = one_e_blocks(
                numpy.ones(ngr)[:, None, None]*Foo[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fov[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fvo[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fvv[None, :, :])
        F = one_e_blocks(Foo, Fov, Fvo, Fvv)
        Fb = Ff
        D1, D2 = test_utils.make_random_ft_D(n)
        tii, gi, Gi = quadrature.simpsons(ngi, beta)
        tir, gr, Gr = quadrature.simpsons(ngr, tmax)

        # compute dL/dt2 from Lagrangian directly
        dT2i = numpy.zeros((ngi, n, n, n, n), dtype=complex)
        dT2b = numpy.zeros((ngr, n, n, n, n), dtype=complex)
        dT2f = numpy.zeros((ngr, n, n, n, n), dtype=complex)
        for y in range(ngi):
            for i in range(n):
                for j in range(n):
                    for a in range(n):
                        for b in range(n):
                            d = 1.e-4
                            TP = T2i.copy()
                            TM = T2i.copy()
                            TP[y, a, b, i, j] += d
                            TP[y, a, b, j, i] -= d
                            TP[y, b, a, i, j] -= d
                            TP[y, b, a, j, i] += d
                            TM[y, a, b, i, j] -= d
                            TM[y, a, b, j, i] += d
                            TM[y, b, a, i, j] += d
                            TM[y, b, a, j, i] -= d
                            LP = evalL(
                                T1f, T1b, T1i, T2f, T2b, TP,
                                L1f, L1b, L1i, L2f, L2b, L2i,
                                Ff, Fb, F, I, D1, D2, tir, tii,
                                gr, gi, Gr, Gi, beta)
                            LM = evalL(
                                T1f, T1b, T1i, T2f, T2b, TM,
                                L1f, L1b, L1i, L2f, L2b, L2i,
                                Ff, Fb, F, I, D1, D2, tir, tii,
                                gr, gi, Gr, Gi, beta)
                            dT2i[y, i, j, a, b] = (LP - LM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for j in range(n):
                    for a in range(n):
                        for b in range(n):
                            d = 1.e-4
                            TP = T2b.copy()
                            TM = T2b.copy()
                            TP[y, a, b, i, j] += d
                            TP[y, a, b, j, i] -= d
                            TP[y, b, a, i, j] -= d
                            TP[y, b, a, j, i] += d
                            TM[y, a, b, i, j] -= d
                            TM[y, a, b, j, i] += d
                            TM[y, b, a, i, j] += d
                            TM[y, b, a, j, i] -= d
                            LP = evalL(
                                T1f, T1b, T1i, T2f, TP, T2i,
                                L1f, L1b, L1i, L2f, L2b, L2i,
                                Ff, Fb, F, I, D1, D2, tir, tii,
                                gr, gi, Gr, Gi, beta)
                            LM = evalL(
                                T1f, T1b, T1i, T2f, TM, T2i,
                                L1f, L1b, L1i, L2f, L2b, L2i,
                                Ff, Fb, F, I, D1, D2, tir, tii,
                                gr, gi, Gr, Gi, beta)
                            dT2b[y, i, j, a, b] = (LP - LM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for j in range(n):
                    for a in range(n):
                        for b in range(n):
                            d = 1.e-4
                            TP = T2f.copy()
                            TM = T2f.copy()
                            TP[y, a, b, i, j] += d
                            TP[y, a, b, j, i] -= d
                            TP[y, b, a, i, j] -= d
                            TP[y, b, a, j, i] += d
                            TM[y, a, b, i, j] -= d
                            TM[y, a, b, j, i] += d
                            TM[y, b, a, i, j] += d
                            TM[y, b, a, j, i] -= d
                            LP = evalL(
                                T1f, T1b, T1i, TP, T2b, T2i,
                                L1f, L1b, L1i, L2f, L2b, L2i,
                                Ff, Fb, F, I, D1, D2, tir, tii,
                                gr, gi, Gr, Gi, beta)
                            LM = evalL(
                                T1f, T1b, T1i, TM, T2b, T2i,
                                L1f, L1b, L1i, L2f, L2b, L2i,
                                Ff, Fb, F, I, D1, D2, tir, tii,
                                gr, gi, Gr, Gi, beta)
                            dT2f[y, i, j, a, b] = (LP - LM)/(2*d)

        # compute dL/dt1 from the Lambda equations
        L1fn, L1bn, L1in, L2fn, L2bn, L2in = \
            ft_cc_equations.neq_lambda_simple(
                    Ff, Fb, F, I, L1f, L1b, L1i, L2f, L2b, L2i, T1f, T1b, T1i,
                    T2f, T2b, T2i, D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi)

        L2in -= L2i
        L2bn -= L2b
        L2fn -= L2f
        for y in range(ngi):
            L2in[y] *= gi[y]/beta
        for y in range(ngr):
            L2bn[y] *= -1.j*gr[y]/beta
            L2fn[y] *= 1.j*gr[y]/beta

        diffi = numpy.linalg.norm(L2in - dT2i)
        diffb = numpy.linalg.norm(L2bn - dT2b)
        difff = numpy.linalg.norm(L2fn - dT2f)
        self.assertTrue(
            abs(diffi) < self.thresh, "Error in dTi: {}".format(diffi))
        self.assertTrue(
            abs(diffb) < self.thresh, "Error in dTb: {}".format(diffb))
        self.assertTrue(
            abs(difff) < self.thresh, "Error in dTf: {}".format(difff))

    def test_lambda_opt(self):
        ngr = 4
        ngi = 4
        n = 3
        beta = 2.0
        tmax = 0.1
        tii, gi, Gi = quadrature.simpsons(ngi, beta)
        tir, gr, Gr = quadrature.simpsons(ngr, tmax)
        T1f, T2f = test_utils.make_random_ft_T(ngr, n)
        T1b, T2b = test_utils.make_random_ft_T(ngr, n)
        T1i, T2i = test_utils.make_random_ft_T(ngi, n)
        L1f, L2f = test_utils.make_random_ft_T(ngr, n)
        L1b, L2b = test_utils.make_random_ft_T(ngr, n)
        L1i, L2i = test_utils.make_random_ft_T(ngi, n)
        T1f = T1f.astype(complex)
        T1b = T1b.astype(complex)
        T2f = T2f.astype(complex)
        T2b = T2b.astype(complex)
        L1f = L1f.astype(complex)
        L1b = L1b.astype(complex)
        L2f = L2f.astype(complex)
        L2b = L2b.astype(complex)
        F, I = test_utils.make_random_integrals(n, n)
        I.vvvv = I.vvvv.astype(complex)
        I.oovv = I.oovv.astype(complex)
        I.vvoo = I.vvoo.astype(complex)
        I.vovo = I.vovo.astype(complex)
        I.oooo = I.oooo.astype(complex)
        Foo = F.oo.astype(complex)
        Fov = F.ov.astype(complex)
        Fvo = F.vo.astype(complex)
        Fvv = F.vv.astype(complex)
        Ff = one_e_blocks(
                numpy.ones(ngr)[:, None, None]*Foo[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fov[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fvo[None, :, :],
                numpy.ones(ngr)[:, None, None]*Fvv[None, :, :])
        F = one_e_blocks(Foo, Fov, Fvo, Fvv)
        Fb = Ff
        D1, D2 = test_utils.make_random_ft_D(n)

        L1fs, L1bs, L1is, L2fs, L2bs, L2is = \
            ft_cc_equations.neq_lambda_simple(
                    Ff, Fb, F, I, L1f, L1b, L1i, L2f, L2b, L2i, T1f, T1b, T1i,
                    T2f, T2b, T2i, D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi)

        L1fo, L1bo, L1io, L2fo, L2bo, L2io = \
            ft_cc_equations.neq_lambda_opt(
                    Ff, Fb, F, I, L1f, L1b, L1i, L2f, L2b, L2i, T1f, T1b, T1i,
                    T2f, T2b, T2i, D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi)

        diff2i = numpy.linalg.norm(L2is - L2io)
        diff2b = numpy.linalg.norm(L2bs - L2bo)
        diff2f = numpy.linalg.norm(L2fs - L2fo)
        diff1i = numpy.linalg.norm(L1is - L1io)
        diff1b = numpy.linalg.norm(L1bs - L1bo)
        diff1f = numpy.linalg.norm(L1fs - L1fo)
        self.assertTrue(
            abs(diff1i) < self.thresh, "Error in T1i: {}".format(diff1i))
        self.assertTrue(
            abs(diff1b) < self.thresh, "Error in T1b: {}".format(diff1b))
        self.assertTrue(
            abs(diff1f) < self.thresh, "Error in T1f: {}".format(diff1f))
        self.assertTrue(
            abs(diff2i) < self.thresh, "Error in T2i: {}".format(diff2i))
        self.assertTrue(
            abs(diff2b) < self.thresh, "Error in T2b: {}".format(diff2b))
        self.assertTrue(
            abs(diff2f) < self.thresh, "Error in T2f: {}".format(diff2f))


if __name__ == '__main__':
    unittest.main()
