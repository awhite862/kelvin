import unittest
import numpy
from cqcpy import test_utils
from kelvin import quadrature

def func(t):
    return t - 0.2*t*t + numpy.sin(t)

def func2(t):
    #return t - 0.2*t*t + numpy.sin(t)
    return numpy.exp(-0.5*t )*0.2*t*t

def make_t(T1,tr):
    nt = T1.shape[0]
    for i in range(nt):
        T1[i] = T1[nt - 1]*func(tr[i])

def make_l(L1,tr):
    nt = L1.shape[0]
    for i in range(nt):
        L1[i] = L1[nt - 1]*func2(tr[i])

class QuadTest(unittest.TestCase):
    def setUp(self):
        self.beta = 2.0
        self.thresh = 1e-7

    def test_int_keldysh(self):
        n = 5
        ngr = 100
        ngi = 50
        tmax = 1.0
        D1,D2 = test_utils.make_random_ft_D(n)
        T1f,T2f = test_utils.make_random_ft_T(ngr,n)
        T1i,T2i = test_utils.make_random_ft_T(ngi,n)
        tir,gr,Gr = quadrature.simpsons(ngr, tmax)
        tii,gi,Gi = quadrature.simpsons(ngi, self.beta)
        make_t(T1f, tir)
        make_t(T2f, tir)
        T1b = T1f.copy()
        T2b = T2f.copy()
        for y in range(ngr):
            T1b[y] = T1f[ngr - y - 1]
            T2b[y] = T2f[ngr - y - 1]
        make_t(T1i, tii)
        make_t(T2i, tii)
        t1ref = quadrature.int_tbar1(ngi,T1i,tii,D1,Gi)
        t2ref = quadrature.int_tbar2(ngi,T2i,tii,D2,Gi)
        t1f,t1b,t1i = quadrature.int_tbar1_keldysh(ngr,ngi,T1f,T1b,T1i,tir,tii,D1,Gr,Gi)
        t2f,t2b,t2i = quadrature.int_tbar2_keldysh(ngr,ngi,T2f,T2b,T2i,tir,tii,D2,Gr,Gi)

        diff1 = numpy.linalg.norm(t1i - t1ref) / t1i.size
        diff2 = numpy.linalg.norm(t2i - t2ref) / t2i.size
        s1 = diff1 < self.thresh
        s2 = diff2 < self.thresh
        self.assertTrue(s1,"Difference in T1: {}".format(diff1))
        self.assertTrue(s2,"Difference in T2: {}".format(diff2))

    def test_Lint_keldysh(self):
        n = 5
        ngr = 32
        ngi = 10
        tmax = 0.5
        D1,D2 = test_utils.make_random_ft_D(n)
        L1f,L2f = test_utils.make_random_ft_T(ngr,n)
        L1i,L2i = test_utils.make_random_ft_T(ngi,n)
        tir,gr,Gr = quadrature.simpsons(ngr, tmax)
        tii,gi,Gi = quadrature.simpsons(ngi, self.beta)
        make_l(L1f, tir)
        make_l(L2f, tir)
        L1b = L1f.copy()
        L2b = L2f.copy()
        for y in range(ngr):
            L1b[y] = L1f[ngr - y - 1]
            L2b[y] = L2f[ngr - y - 1]
        make_l(L1i, tii)
        make_l(L2i, tii)
        l1ref = quadrature.int_L1(ngi,L1i,tii,D1,gi,Gi)
        l2ref = quadrature.int_L2(ngi,L2i,tii,D2,gi,Gi)
        l1f,l1b,l1i = quadrature.int_L1_keldysh(ngr,ngi,L1f,L1b,L1i,tir,tii,D1,gr,gi,Gr,Gi)
        l2f,l2b,l2i = quadrature.int_L2_keldysh(ngr,ngi,L2f,L2b,L2i,tir,tii,D2,gr,gi,Gr,Gi)

        diff1 = numpy.linalg.norm(l1i - l1ref) / l1i.size
        diff2 = numpy.linalg.norm(l2i - l2ref) / l2i.size
        s1 = diff1 < self.thresh
        s2 = diff2 < self.thresh
        self.assertTrue(s1,"Difference in L1: {}".format(diff1))

    def test_d_simpson(self):
        ng = 10
        beta = 10.0
        delta = 5e-4
        ti,gp,Gp = quadrature.simpsons(ng, beta + delta)
        ti,gm,Gm = quadrature.simpsons(ng, beta - delta)
        gd,Gd = quadrature.d_simpsons(ng, beta)
        go = (gp - gm)/(2.0*delta)
        Go = (Gp - Gm)/(2.0*delta)
        eg = numpy.linalg.norm(go - gd)
        eG = numpy.linalg.norm(Go - Go)
        self.assertTrue(eg < self.thresh,"Difference in g: {}".format(eg))
        self.assertTrue(eG < self.thresh,"Difference in G: {}".format(eG))

    def test_d_simpson_ln(self):
        ng = 10
        beta = 10.0
        delta = 5e-4
        ti,gp,Gp = quadrature.simpsons_ln(ng, beta + delta)
        ti,gm,Gm = quadrature.simpsons_ln(ng, beta - delta)
        gd,Gd = quadrature.d_simpsons_ln(ng, beta)
        go = (gp - gm)/(2.0*delta)
        Go = (Gp - Gm)/(2.0*delta)
        eg = numpy.linalg.norm(go - gd)
        eG = numpy.linalg.norm(Go - Go)
        self.assertTrue(eg < self.thresh,"Difference in g: {}".format(eg))
        self.assertTrue(eG < self.thresh,"Difference in G: {}".format(eG))

    def test_d_simpson_sin(self):
        ng = 10
        beta = 10.0
        delta = 5e-4
        ti,gp,Gp = quadrature.simpsons_sin(ng, beta + delta)
        ti,gm,Gm = quadrature.simpsons_sin(ng, beta - delta)
        gd,Gd = quadrature.d_simpsons_sin(ng, beta)
        go = (gp - gm)/(2.0*delta)
        Go = (Gp - Gm)/(2.0*delta)
        eg = numpy.linalg.norm(go - gd)
        eG = numpy.linalg.norm(Go - Go)
        self.assertTrue(eg < self.thresh,"Difference in g: {}".format(eg))
        self.assertTrue(eG < self.thresh,"Difference in G: {}".format(eG))

    def test_d_simpson_exp(self):
        ng = 10
        beta = 10.0
        delta = 5e-4
        ti,gp,Gp = quadrature.simpsons_exp(ng, beta + delta)
        ti,gm,Gm = quadrature.simpsons_exp(ng, beta - delta)
        gd,Gd = quadrature.d_simpsons_exp(ng, beta)
        go = (gp - gm)/(2.0*delta)
        Go = (Gp - Gm)/(2.0*delta)
        eg = numpy.linalg.norm(go - gd)
        eG = numpy.linalg.norm(Go - Go)
        self.assertTrue(eg < self.thresh,"Difference in g: {}".format(eg))
        self.assertTrue(eG < self.thresh,"Difference in G: {}".format(eG))

    def test_d_simpson_p(self):
        ng = 10
        beta = 10.0
        delta = 5e-4
        n = 3
        ti,gp,Gp = quadrature.simpsons_p(ng, beta + delta, n)
        ti,gm,Gm = quadrature.simpsons_p(ng, beta - delta, n)
        gd,Gd = quadrature.d_simpsons_p(ng, beta, n)
        go = (gp - gm)/(2.0*delta)
        Go = (Gp - Gm)/(2.0*delta)
        eg = numpy.linalg.norm(go - gd)
        eG = numpy.linalg.norm(Go - Go)
        self.assertTrue(eg < self.thresh,"Difference in g: {}".format(eg))
        self.assertTrue(eG < self.thresh,"Difference in G: {}".format(eG))

if __name__ == '__main__':
    unittest.main()
