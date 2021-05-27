import unittest
import numpy
from cqcpy.utils import block_diag
from kelvin.ccsd import ccsd
from kelvin.neq_ccsd import neq_ccsd
from kelvin.td_ccsd import TDCCSD
from kelvin.kel_ccsd import KelCCSD
try:
    from lattice.hubbard import Hubbard1D
    from lattice.fci import FCISimple
    from kelvin.hubbard_system import HubbardSystem
    from kelvin.hubbard_field_system import hubbard_field_system, HubbardFieldSystem
    has_lattice = True
except ImportError:
    has_lattice = False


def buildH(hub, phase=None):
    dim = 16
    myfci = FCISimple(hub, 1, m_s=1)
    H1 = myfci.getH(phase=phase)
    myfci = FCISimple(hub, 2, m_s=0)
    H20 = myfci.getH(phase=phase)
    myfci = FCISimple(hub, 2, m_s=2)
    H22 = myfci.getH(phase=phase)
    myfci = FCISimple(hub, 3, m_s=1)
    H31 = myfci.getH(phase=phase)
    myfci = FCISimple(hub, 4, m_s=0)
    H4 = myfci.getH(phase=phase)
    H = numpy.zeros((dim, dim), dtype=complex)
    x = 1
    y = 3
    z = 5
    a = 9
    b = 10
    c = 11
    d = 13
    e = 15
    H[x:y, x:y] = H1
    H[y:z, y:z] = H1
    H[z:a, z:a] = H20
    H[a:b, a:b] = H22
    H[b:c, b:c] = H22
    H[c:d, c:d] = H31
    H[d:e, d:e] = H31
    H[e:, e:] = H4
    return H


def matrix_exp(M):
    n, m = M.shape
    assert(n == m)
    U = numpy.identity(m, dtype=complex)
    Mn = M.copy()
    for i in range(1, 13):
        U = U + Mn
        Mn = numpy.einsum('pq,qr->pr', Mn, M)/float(i+1)

    return U


@unittest.skipUnless(has_lattice, "Lattice module cannot be found")
class HubbardFieldTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_null_cc(self):
        A0 = 0.0
        t0 = 1.0
        sigma = 0.25
        omega = 4.4
        tmax = 0.5
        ng = 400
        ngi = 10
        deltat = tmax / ng

        T = 1.0
        mu = 0.0
        L = 2
        U = 1.0
        hub = Hubbard1D(L, 1.0, U, boundary='o')
        ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])

        Pa = numpy.zeros((2, 2))
        Pb = numpy.zeros((2, 2))
        Pa[0, 0] = 1.0
        Pb[1, 1] = 1.0
        sys = hubbard_field_system(T, hub, ti, A0, t0, sigma, omega, mu=mu, Pa=Pa, Pb=Pb)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-9, max_iter=40, damp=0.1, ngr=ng, ngi=ngi, iprint=0)
        E, Ecc = cc.run()

        sys = HubbardSystem(T, hub, Pa, Pb, mu=mu, orbtype='u')
        cc = ccsd(sys, iprint=0, max_iter=80, econv=1e-11, T=T, mu=mu, ngrid=ngi)
        Eoutr, Eccr = cc.run()
        diff = abs(Eoutr - E)
        msg = "Difference: {}".format(diff)
        self.assertTrue(diff < 5e-4, msg)

    def test_cc_vs_fci(self):
        A0 = 0.3
        t0 = 1.0
        sigma = 0.25
        omega = 4.4
        phi = 0.0
        tmax = 0.5
        ng = 400
        ngi = 20
        deltat = tmax / ng

        T = 1.5
        mu = 0.0
        beta = 1.0/T
        L = 2
        U = 1.0
        hub = Hubbard1D(L, 1.0, U, boundary='o')
        ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])

        # 00 u0 0u d0 0d ud0 ud du 0ud uu dd udu uud udd dud udud
        # 0  1  2  3  4  5   6  7  8   9  10 11  12  13  14  15
        dim = 16

        # Construct H, U, E and mz^2
        H = buildH(hub)
        E, U = numpy.linalg.eigh(H)
        N = numpy.zeros((dim, dim))
        N[1, 1] = 1.0
        N[2, 2] = 1.0
        N[3, 3] = 1.0
        N[4, 4] = 1.0
        N[5, 5] = 2.0
        N[6, 6] = 2.0
        N[7, 7] = 2.0
        N[8, 8] = 2.0
        N[9, 9] = 2.0
        N[10, 10] = 2.0
        N[11, 11] = 3.0
        N[12, 12] = 3.0
        N[13, 13] = 3.0
        N[14, 14] = 3.0
        N[15, 15] = 4.0
        M = numpy.zeros((dim, dim))
        M[1, 1] = 1.0
        M[2, 2] = 0.0
        M[3, 3] = 1.0
        M[4, 4] = 0.0
        M[5, 5] = 2.0
        M[6, 6] = 1.0
        M[7, 7] = 1.0
        M[8, 8] = 0.0
        M[9, 9] = 1.0
        M[10, 10] = 1.0
        M[11, 11] = 2.0
        M[12, 12] = 1.0
        M[13, 13] = 2.0
        M[14, 14] = 1.0
        M[15, 15] = 2.0
        Ns = numpy.einsum('ij,ip,jq->pq', N, U, U).diagonal()

        # construct density matrix at t = 0
        exp = numpy.exp(-beta*(E - mu*Ns))
        Z = exp.sum()
        P = numpy.einsum('mi,i,ni->mn', U, exp, U) / Z

        Ms = []
        # propagate density and measure M
        for i, t in enumerate(ti):
            # propagate the density
            dt = t - t0
            ex = dt*dt/(2*sigma*sigma)
            phase = A0*numpy.exp(-ex)*numpy.cos(omega*dt + phi)
            Ht = buildH(hub, phase=phase)
            e, v = numpy.linalg.eigh(Ht)
            ee = numpy.exp(-deltat*1.j*e)
            ee2 = numpy.exp(deltat*1.j*e)
            U = numpy.einsum('ai,i,bi->ab', v, ee, numpy.conj(v))
            U2 = numpy.einsum('ai,i,bi->ab', v, ee2, numpy.conj(v))
            P = numpy.einsum('sp,pq,qr->sr', U, P, U2)

            # measure M
            Ms.append(numpy.einsum('ij,ji->', P, M))

        Pa = numpy.zeros((2, 2))
        Pb = numpy.zeros((2, 2))
        Pa[0, 0] = 1.0
        Pb[1, 1] = 1.0
        sys = hubbard_field_system(T, hub, ti, A0, t0, sigma, omega, mu=mu, Pa=Pa, Pb=Pb)
        m = numpy.zeros((2, 2))
        m[0, 0] = 1.0
        ma = numpy.einsum('ij,ip,jq->pq', m, sys.ua, sys.ua)
        mb = numpy.einsum('ij,ip,jq->pq', m, sys.ub, sys.ub)
        mg = block_diag(ma, mb)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-9, max_iter=40, damp=0.1, ngr=ng, ngi=ngi, iprint=0)
        E, Ecc = cc.run()
        cc._neq_ccsd_lambda()
        cc._neq_1rdm()
        Mscc = []
        for i, t in enumerate(ti):
            out = cc.compute_prop(mg, i)
            Mscc.append(out)

        for i, m in enumerate(Ms):
            diff = abs(m - Mscc[i])
            msg = "{}: {} {}".format(i, m, Mscc[i])
            self.assertTrue(diff < 2e-4, msg)

    def test_kel_cc(self):
        A0 = 0.3
        t0 = 1.0
        sigma = 0.25
        omega = 4.4
        phi = 0.0
        tmax = 0.5
        ng = 400
        ngi = 80
        deltat = tmax / ng

        T = 1.5
        mu = 0.0
        beta = 1.0/T
        L = 2
        U = 1.0
        hub = Hubbard1D(L, 1.0, U, boundary='o')
        ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])

        # 00 u0 0u d0 0d ud0 ud du 0ud uu dd udu uud udd dud udud
        # 0  1  2  3  4  5   6  7  8   9  10 11  12  13  14  15
        dim = 16

        # Construct H, U, E and mz^2
        H = buildH(hub)
        E, U = numpy.linalg.eigh(H)
        N = numpy.zeros((dim, dim))
        N[1, 1] = 1.0
        N[2, 2] = 1.0
        N[3, 3] = 1.0
        N[4, 4] = 1.0
        N[5, 5] = 2.0
        N[6, 6] = 2.0
        N[7, 7] = 2.0
        N[8, 8] = 2.0
        N[9, 9] = 2.0
        N[10, 10] = 2.0
        N[11, 11] = 3.0
        N[12, 12] = 3.0
        N[13, 13] = 3.0
        N[14, 14] = 3.0
        N[15, 15] = 4.0
        M = numpy.zeros((dim, dim))
        M[1, 1] = 1.0
        M[2, 2] = 0.0
        M[3, 3] = 1.0
        M[4, 4] = 0.0
        M[5, 5] = 2.0
        M[6, 6] = 1.0
        M[7, 7] = 1.0
        M[8, 8] = 0.0
        M[9, 9] = 1.0
        M[10, 10] = 1.0
        M[11, 11] = 2.0
        M[12, 12] = 1.0
        M[13, 13] = 2.0
        M[14, 14] = 1.0
        M[15, 15] = 2.0
        Ns = numpy.einsum('ij,ip,jq->pq', N, U, U).diagonal()

        # construct density matrix at t = 0
        exp = numpy.exp(-beta*(E - mu*Ns))
        Z = exp.sum()
        P = numpy.einsum('mi,i,ni->mn', U, exp, U)/Z

        Ms = [numpy.einsum('ij,ji->', P, M)]
        # propagate density and measure M
        for i, t in enumerate(ti):
            # propagate the density
            dt = (i)*deltat - t0
            ex = dt*dt/(2*sigma*sigma)
            phase = A0*numpy.exp(-ex)*numpy.cos(omega*dt + phi)
            Ht = buildH(hub, phase=phase)
            e, v = numpy.linalg.eigh(Ht)
            ee = numpy.exp(-deltat*1.j*e)
            ee2 = numpy.exp(deltat*1.j*e)
            U = numpy.einsum('ai,i,bi->ab', v, ee, numpy.conj(v))
            U2 = numpy.einsum('ai,i,bi->ab', v, ee2, numpy.conj(v))
            P = numpy.einsum('sp,pq,qr->sr', U, P, U2)

            # measure M
            Ms.append(numpy.einsum('ij,ji->', P, M))

        Pa = numpy.zeros((2, 2))
        Pb = numpy.zeros((2, 2))
        Pa[0, 0] = 1.0
        Pb[1, 1] = 1.0
        sys = hubbard_field_system(T, hub, ti, A0, t0, sigma, omega, mu=mu, Pa=Pa, Pb=Pb)
        m = numpy.zeros((2, 2))
        m[0, 0] = 1.0
        ma = numpy.einsum('ij,ip,jq->pq', m, sys.ua, sys.ua)
        mb = numpy.einsum('ij,ip,jq->pq', m, sys.ub, sys.ub)
        mg = block_diag(ma, mb)

        Mscc2 = []
        sys = HubbardFieldSystem(T, hub, A0, t0, sigma, omega, mu=mu, Pa=Pa, Pb=Pb)
        prop = {"tprop": "rk4", "lprop": "rk4"}
        mycc = TDCCSD(sys, prop, T=T, mu=mu, iprint=0, ngrid=ngi, saveT=True, saveL=True)
        mycc.run()
        mycc._ccsd_lambda()

        prop = {"tprop": "rk1", "lprop": "rk1"}
        kccsd = KelCCSD(sys, prop, T=T, mu=mu, iprint=0)
        kccsd.init_from_ftccsd(mycc, contour="keldysh")
        kccsd._ccsd(nstep=400, step=0.00125)
        for i, p in enumerate(kccsd.P):
            Mscc2.append(numpy.einsum('ij,ji->', mg, p))

        for i, ms in enumerate(zip(Ms, Mscc2)):
            m, m2 = ms
            diff = abs(m - m2)
            msg = "{}: {} {}".format(i, m, m2)
            self.assertTrue(diff < 2e-4, msg)


if __name__ == '__main__':
    unittest.main()
