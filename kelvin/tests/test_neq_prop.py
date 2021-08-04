import unittest
import numpy
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy import integrals
from kelvin import cc_utils
from kelvin import ft_cc_energy
from kelvin import ft_cc_equations
from kelvin import quadrature
from kelvin.h2_field_system import h2_field_system
from kelvin.neq_ccsd import neq_ccsd


def evalLd(T1f, T1b, T1i, T2f, T2b, T2i, L1f, L1b, L1i, L2f, L2b, L2i,
           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta):

    ngr = gr.shape[0]
    ngi = gi.shape[0]
    E = ft_cc_energy.ft_cc_energy_neq(
        T1f, T1b, T1i, T2f, T2b, T2i,
        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
    T1f_, T1b_, T1i_, T2f_, T2b_, T2i_ =\
        ft_cc_equations.neq_ccsd_simple(
            Ff, Fb, F, I, T1f, T1b, T1i, T2f, T2b, T2i,
            D1, D2, tir, tii, ngr, ngi, Gr, Gi)
    TEf = 0.25*numpy.einsum('yijab,yabij->y', L2f, T2f_)
    TEf += numpy.einsum('yia,yai->y', L1f, T1f_)
    TEb = 0.25*numpy.einsum('yijab,yabij->y', L2b, T2b_)
    TEb += numpy.einsum('yia,yai->y', L1b, T1b_)
    TEi = 0.25*numpy.einsum('yijab,yabij->y', L2i, T2i_)
    TEi += numpy.einsum('yia,yai->y', L1i, T1i_)

    Te = (1.j/beta)*numpy.einsum('y,y->', TEf, gr)
    Te -= (1.j/beta)*numpy.einsum('y,y->', TEb, gr)
    Te += (1.0/beta)*numpy.einsum('y,y->', TEi, gi)

    return E + Te


class NEQPropTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 2e-4
        self.fd_thresh = 1e-8

    def test_h2_field_deriv(self):
        beta = 1.0
        T = 1./beta
        mu = 0.0
        omega = 0.5
        ngi = 2
        n = 2

        mol = gto.M(
            verbose=0,
            atom='H 0 0 -0.6; H 0 0 0.0',
            basis='STO-3G',
            charge=1,
            spin=1)

        m = scf.UHF(mol)
        m.scf()
        mos = m.mo_coeff[0]
        E = numpy.zeros((3))
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq', mos, field, mos)

        # compute property from finite differences
        deltat = 0.5
        tmax = 1.0
        ng = int(tmax/deltat)
        ngr = ng
        tii, gi, Gi = quadrature.simpsons(ngi, beta)
        tir, gr, Gr = quadrature.midpoint(ngr, tmax)
        d = 1e-4
        sys = h2_field_system(T, mu, omega, tir, O=(d*field), ot=ng - 1)
        en = sys.g_energies_tot()
        F_f, Ff_f, Fb_f, I_f = cc_utils.get_ft_integrals_neq(sys, en, beta, mu)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-12,
                      max_iter=40, damp=0.0, ngr=ng, ngi=ngi, iprint=0)
        Ef = cc.run()
        t1fp = cc.T1f
        t1bp = cc.T1b
        t1ip = cc.T1i
        t2fp = cc.T2f
        t2bp = cc.T2b
        t2ip = cc.T2i

        sys = h2_field_system(T, mu, omega, tir, O=(-d*field), ot=ng - 1)
        en = sys.g_energies_tot()
        F_b, Ff_b, Fb_b, I_b = cc_utils.get_ft_integrals_neq(sys, en, beta, mu)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-12,
                      max_iter=40, damp=0.0, ngr=ng, ngi=ngi, iprint=0)
        Eb = cc.run()
        t1fm = cc.T1f
        t1bm = cc.T1b
        t1im = cc.T1i
        t2fm = cc.T2f
        t2bm = cc.T2b
        t2im = cc.T2i
        Efd = (Ef[0] - Eb[0])/(2*d)

        # derivatives of amplitudes from FD
        td1f = (t1fp - t1fm)/(2*d)
        td1b = (t1bp - t1bm)/(2*d)
        td1i = (t1ip - t1im)/(2*d)
        td2f = (t2fp - t2fm)/(2*d)
        td2b = (t2bp - t2bm)/(2*d)
        td2i = (t2ip - t2im)/(2*d)

        # unperturbed Neq-CCSD
        sys = h2_field_system(T, mu, omega, tir, O=None, ot=None)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-12,
                      max_iter=40, damp=0.0, ngr=ng, ngi=ngi, iprint=0)
        cc.run()
        cc._neq_ccsd_lambda()
        cc._neq_1rdm()

        # energies and occupations
        en = sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get energy differences
        D1 = utils.D1(en, en)
        D2 = utils.D2(en, en)

        # compute first order part
        E1 = numpy.einsum('ii,i->', field, fo)

        Ef = ft_cc_energy.ft_cc_energy_neq(
            cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
            Ff_f.ov, Fb_f.ov, F_f.ov, I_f.oovv, gr, gi, beta)
        Eb = ft_cc_energy.ft_cc_energy_neq(
            cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
            Ff_b.ov, Fb_b.ov, F_b.ov, I_b.oovv, gr, gi, beta)

        Esfd = (Ef - Eb)/(2*d)

        # integrals
        Fock = numpy.zeros((ng, n, n), dtype=complex)
        Fock[ng - 1] -= 1.j*beta*field/deltat
        Foo = numpy.einsum('yij,j->yij', Fock, fo)
        Fvo = numpy.einsum('yai,a,i->yai', Fock, fv, fo)
        Fvv = numpy.einsum('yab,a->yab', Fock, fv)
        temp = numpy.zeros((ng, n, n), dtype=complex)
        Ffn = one_e_blocks(Foo, Fock, Fvo, Fvv)
        Fbn = one_e_blocks(temp, temp, temp, temp)
        Fn = one_e_blocks(Fock[0], Fock[0], Fock[0], Fock[0])
        Inull = numpy.zeros((n, n, n, n), dtype=complex)
        In = two_e_blocks(
            vvvv=Inull, vvvo=Inull, vovv=Inull,
            vvoo=Inull, vovo=Inull, oovv=Inull,
            vooo=Inull, ooov=Inull, oooo=Inull)

        # compute the static part directly
        Es = ft_cc_energy.ft_cc_energy_neq(
            cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
            Ffn.ov, Fbn.ov, Fn.ov, In.oovv, gr, gi, beta)

        # get actual integrals
        F, Ff, Fb, I = cc_utils.get_ft_integrals_neq(sys, en, beta, mu)

        # compute the singles
        dT1i = numpy.zeros((ngi, n, n), dtype=complex)
        dT1b = numpy.zeros((ngr, n, n), dtype=complex)
        dT1f = numpy.zeros((ngr, n, n), dtype=complex)
        for y in range(ngi):
            for i in range(n):
                for a in range(n):
                    d = 2.e-4
                    TP = cc.T1i.copy()
                    TM = cc.T1i.copy()
                    TP[y, a, i] += d
                    TM[y, a, i] -= d
                    EP = ft_cc_energy.ft_cc_energy_neq(
                        cc.T1f, cc.T1b, TP, cc.T2f, cc.T2b, cc.T2i,
                        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                    EM = ft_cc_energy.ft_cc_energy_neq(
                        cc.T1f, cc.T1b, TM, cc.T2f, cc.T2b, cc.T2i,
                        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                    dT1i[y, i, a] = (EP - EM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for a in range(n):
                    d = 2.e-4
                    TP = cc.T1b.copy()
                    TM = cc.T1b.copy()
                    TP[y, a, i] += d
                    TM[y, a, i] -= d
                    EP = ft_cc_energy.ft_cc_energy_neq(
                        cc.T1f, TP, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                    EM = ft_cc_energy.ft_cc_energy_neq(
                        cc.T1f, TM, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                    dT1b[y, i, a] = (EP - EM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for a in range(n):
                    d = 2.e-4
                    TP = cc.T1f.copy()
                    TM = cc.T1f.copy()
                    TP[y, a, i] += d
                    TM[y, a, i] -= d
                    EP = ft_cc_energy.ft_cc_energy_neq(
                        TP, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                    EM = ft_cc_energy.ft_cc_energy_neq(
                        TM, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                    dT1f[y, i, a] = (EP - EM)/(2*d)

        dT2i = numpy.zeros((ngi, n, n, n, n), dtype=complex)
        dT2b = numpy.zeros((ngr, n, n, n, n), dtype=complex)
        dT2f = numpy.zeros((ngr, n, n, n, n), dtype=complex)
        for y in range(ngi):
            for i in range(n):
                for j in range(n):
                    for a in range(n):
                        for b in range(n):
                            d = 2.e-4
                            TP = cc.T2i.copy()
                            TM = cc.T2i.copy()
                            TP[y, a, b, i, j] += d
                            TP[y, a, b, j, i] -= d
                            TP[y, b, a, i, j] -= d
                            TP[y, b, a, j, i] += d
                            TM[y, a, b, i, j] -= d
                            TM[y, a, b, j, i] += d
                            TM[y, b, a, i, j] += d
                            TM[y, b, a, j, i] -= d
                            EP = ft_cc_energy.ft_cc_energy_neq(
                                cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, TP,
                                Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                            EM = ft_cc_energy.ft_cc_energy_neq(
                                cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, TM,
                                Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                            dT2i[y, i, j, a, b] = (EP - EM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for j in range(n):
                    for a in range(n):
                        for b in range(n):
                            d = 2.e-4
                            TP = cc.T2b.copy()
                            TM = cc.T2b.copy()
                            TP[y, a, b, i, j] += d
                            TP[y, a, b, j, i] -= d
                            TP[y, b, a, i, j] -= d
                            TP[y, b, a, j, i] += d
                            TM[y, a, b, i, j] -= d
                            TM[y, a, b, j, i] += d
                            TM[y, b, a, i, j] += d
                            TM[y, b, a, j, i] -= d
                            EP = ft_cc_energy.ft_cc_energy_neq(
                                cc.T1f, cc.T1b, cc.T1i, cc.T2f, TP, cc.T2i,
                                Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                            EM = ft_cc_energy.ft_cc_energy_neq(
                                cc.T1f, cc.T1b, cc.T1i, cc.T2f, TM, cc.T2i,
                                Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                            dT2b[y, i, j, a, b] = (EP - EM)/(2*d)
        for y in range(ngr):
            for i in range(n):
                for j in range(n):
                    for a in range(n):
                        for b in range(n):
                            d = 2.e-4
                            TP = cc.T2f.copy()
                            TM = cc.T2f.copy()
                            TP[y, a, b, i, j] += d
                            TP[y, a, b, j, i] -= d
                            TP[y, b, a, i, j] -= d
                            TP[y, b, a, j, i] += d
                            TM[y, a, b, i, j] -= d
                            TM[y, a, b, j, i] += d
                            TM[y, b, a, i, j] += d
                            TM[y, b, a, j, i] -= d
                            EP = ft_cc_energy.ft_cc_energy_neq(
                                cc.T1f, cc.T1b, cc.T1i, TP, cc.T2b, cc.T2i,
                                Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                            EM = ft_cc_energy.ft_cc_energy_neq(
                                cc.T1f, cc.T1b, cc.T1i, TM, cc.T2b, cc.T2i,
                                Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
                            dT2f[y, i, j, a, b] = (EP - EM)/(2*d)
        Ers1 = numpy.einsum('yia,yai->', dT1f, td1f)
        Ers1 += numpy.einsum('yia,yai->', dT1b, td1b)
        Ers1 += numpy.einsum('yia,yai->', dT1i, td1i)
        Ers2 = 0.25*numpy.einsum('yijab,yabij->', dT2f, td2f)
        Ers2 += 0.25*numpy.einsum('yijab,yabij->', dT2b, td2b)
        Ers2 += 0.25*numpy.einsum('yijab,yabij->', dT2i, td2i)

        Ds = abs(Es - Esfd)
        Dd = abs(Ers1 + Ers2 - Efd + E1 + Esfd)
        self.assertTrue(
            Ds < self.fd_thresh, "Error in energy response: {}".format(Ds))
        self.assertTrue(
            Dd < self.fd_thresh, "Error in amplitude response: {}".format(Dd))

        # evaluate Lagrangian partial and confirm it is equal to the
        # total derivative
        L = evalLd(cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                   cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                   Ffn, Fbn, Fn, In, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
        Dl = abs(L - (Efd - E1))
        self.assertTrue(
            Dl < self.fd_thresh, "Error in lagrangian response: {}".format(Dl))

        # evaluate with the density
        out = cc.compute_prop(field, ngr - 1)
        Dp = abs(out - L - E1)
        self.assertTrue(
            Dp < self.fd_thresh, "Error in response density: {}".format(Dp))

    def test_h2_field(self):
        beta = 1.0
        T = 1./beta
        mu = 0.0
        omega = 0.5

        mol = gto.M(
            verbose=0,
            atom='H 0 0 -0.6; H 0 0 0.0',
            basis='STO-3G',
            charge=1,
            spin=1)

        m = scf.UHF(mol)
        m.scf()
        mos = m.mo_coeff[0]

        eri = integrals.get_phys(mol, mos, mos, mos, mos)
        hcore = numpy.einsum('mp,mn,nq->pq', mos, m.get_hcore(m.mol), mos)
        F = hcore + eri[:, 0, :, 0] - eri[:, 0, 0, :]
        en, vvvvv = numpy.linalg.eigh(F)

        E = numpy.zeros((3))
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq', mos, field, mos)

        # Neq-CCSD reference with FD
        Aref = []
        deltat = 0.01
        for i in range(9):
            tmax = 9*0.1
            ng = int(tmax/deltat)
            tf = (i+1)*0.1
            gf = int(tf/deltat)
            d = 2e-4
            ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])
            PT = field
            sys = h2_field_system(T, mu, omega, ti, O=(d*PT), ot=gf - 1)
            cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-10,
                          max_iter=40, damp=0.0, ngr=ng, ngi=40, iprint=0)
            Ef = cc.run()

            sys = h2_field_system(T, mu, omega, ti, O=(-d*PT), ot=gf - 1)
            cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-10,
                          max_iter=40, damp=0.0, ngr=ng, ngi=40, iprint=0)
            Eb = cc.run()
            Aref.append((Ef[0] - Eb[0])/(2*d))

        # Neq-CCSD from density
        tmax = 0.9
        ng = int(tmax/deltat)
        ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])
        sys = h2_field_system(T, mu, omega, ti, O=None, ot=None)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-10,
                      max_iter=40, damp=0.0, ngr=ng, ngi=40, iprint=0)
        cc.run()
        cc._neq_ccsd_lambda()
        cc._neq_1rdm()

        for i, ref in enumerate(Aref):
            t = (i+1)*int(0.1/deltat)
            out = cc.compute_prop(field, t - 1)
            diff = abs(ref - out)
            msg = "{} -- Expected: {}  Actual: {} ".format(i, ref, out)
            self.assertTrue(diff < self.fd_thresh, msg)

    def test_h2_field2(self):
        beta = 2.0
        T = 1./beta
        mu = 0.0
        omega = 0.5

        ngrid_ref = 4000
        deltat = 0.00025
        mol = gto.M(
            verbose=0,
            atom='H 0 0 -0.6; H 0 0 0.0',
            basis='STO-3G',
            charge=1,
            spin=1)

        m = scf.UHF(mol)
        m.scf()
        mos = m.mo_coeff[0]

        eri = integrals.get_phys(mol, mos, mos, mos, mos)
        hcore = numpy.einsum('mp,mn,nq->pq', mos, m.get_hcore(m.mol), mos)

        E = numpy.zeros((3))
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq', mos, field, mos)
        H = numpy.zeros((4, 4))
        H[1, 1] += hcore[0, 0]
        H[2, 2] += hcore[1, 1]
        H[1, 2] += hcore[0, 1]
        H[2, 1] += hcore[1, 0]
        H[3, 3] = hcore[0, 0] + hcore[1, 1] + eri[0, 1, 0, 1] - eri[0, 1, 1, 0]
        Hint = numpy.zeros((4, 4))
        Hint[1, 1] = field[0, 0]
        Hint[2, 2] = field[1, 1]
        Hint[1, 2] = field[0, 1]
        Hint[2, 1] = field[1, 0]
        Hint[3, 3] = field[0, 0] + field[1, 1]

        e0, v0 = numpy.linalg.eigh(H)
        exp = numpy.exp(-beta*(e0))
        Z = exp.sum()
        p0 = numpy.einsum('mi,i,ni->mn', v0, exp, v0) / Z
        p = p0.copy()
        ti = numpy.zeros(ngrid_ref)
        Aref = []

        for i in range(ngrid_ref):
            t = i*deltat
            ti[i] = t
            Ht = H + numpy.sin(omega*t)*Hint
            e, v = numpy.linalg.eigh(Ht)
            ee = numpy.exp(-deltat*1.j*e)
            ee2 = numpy.exp(deltat*1.j*e)
            U = numpy.einsum('ai,i,bi->ab', v, ee, numpy.conj(v))
            U2 = numpy.einsum('ai,i,bi->ab', v, ee2, numpy.conj(v))
            p = numpy.einsum('sp,pq,qr->sr', U, p, U2)
            if i % (ngrid_ref//10) == 0:
                Aref.append(numpy.einsum('ij,ji->', p, Hint))

        # Neq-CCSD from density
        tmax = 0.9
        deltat = 0.0005
        ng = int(tmax/deltat)
        ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])
        sys = h2_field_system(T, mu, omega, ti, O=None, ot=None)
        cc = neq_ccsd(sys, T, mu=mu, tmax=tmax, econv=1e-10,
                      max_iter=40, damp=0.0, ngr=ng, ngi=40, iprint=0)
        cc.run()
        cc._neq_ccsd_lambda()
        cc._neq_1rdm()

        for i, ref in enumerate(Aref):
            t = i*int(0.1/deltat)
            tg = t if i == 0 else t - 1
            out = cc.compute_prop(field, tg)
            diff = abs(ref - out)
            msg = "{} -- Expected: {}  Actual: {} ".format(i, ref, out)
            self.assertTrue(diff < self.thresh, msg)

    #def test_h2_field(self):
    #    beta = 1.0
    #    T = 1./beta
    #    mu = 0.0
    #    omega = 0.5

    #    #ngrid_ref = 4000
    #    #deltat = 0.00025
    #    mol = gto.M(
    #        verbose = 0,
    #        atom = 'H 0 0 -0.6; H 0 0 0.0',
    #        basis = 'STO-3G',
    #        charge = 1,
    #        spin = 1)

    #    m = scf.UHF(mol)
    #    m.scf()
    #    mos = m.mo_coeff[0]

    #    eri = integrals.get_phys(mol, mos, mos, mos, mos)
    #    hcore = numpy.einsum('mp,mn,nq->pq',mos,m.get_hcore(m.mol),mos)
    #    F = hcore + eri[:,0,:,0] - eri[:,0,0,:]
    #    en,vvvvv = numpy.linalg.eigh(F)

    #    E = numpy.zeros((3))
    #    E[2] = 1.0
    #    field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
    #    field = numpy.einsum('mp,mn,nq->pq',mos,field,mos)

    #    # Neq-CCSD reference with FD
    #    ngi = 10
    #    Aref = []
    #    deltat = 0.001
    #    for i in range(6,9):
    #        tmax = (i)*0.1
    #        ng = int(tmax/deltat) - 1
    #        tf = 6*0.1
    #        gf = int(tf/deltat) - 1
    #        print(gf - 1)
    #        d = 1e-2
    #        ti = numpy.asarray(
    #            [float(j)*deltat  + deltat/2 for j in range(ng)])
    #        D1 = en[:,None] - en[None,:]
    #        dt = (tmax - tf)
    #        #PT = field*numpy.exp(-1.j*D1*dt)
    #        fac = 1.0 #if i == 6 else 0.5
    #        PT = fac*field
    #        sys = h2_field_system(T,mu,omega,ti,O=(d*PT),ot=gf - 1)
    #        cc = neq_ccsd(sys,T,mu=mu,tmax=tmax,econv=1e-10,max_iter=40,damp=0.0,ngr=ng,ngi=40,iprint=0)
    #        Ef = cc.run()

    #        sys = h2_field_system(T,mu,omega,ti,O=(-d*PT),ot=gf - 1)
    #        cc = neq_ccsd(sys,T,mu=mu,tmax=tmax,econv=1e-10,max_iter=40,damp=0.0,ngr=ng,ngi=40,iprint=0)
    #        Eb = cc.run()
    #        print(Eb[1],Ef[1])
    #        print((Ef[0] - Eb[0])/(2*d))
    #        print("\n")
    #        Aref.append((Ef[0] - Eb[0])/(2*d))

    #    # Neq-CCSD from density
    #    tmax = 0.6
    #    ng = int(tmax/deltat) - 1
    #    #ti = numpy.asarray([float(j)*deltat for j in range(ng)])
    #    ti = numpy.asarray([float(j)*deltat  + deltat/2 for j in range(ng)])
    #    sys = h2_field_system(T,mu,omega,ti,O=None,ot=None)
    #    cc = neq_ccsd(sys,T,mu=mu,tmax=tmax,econv=1e-10,max_iter=40,damp=0.0,ngr=ng,ngi=40,iprint=1)
    #    cc.run()
    #    cc._neq_ccsd_lambda()
    #    cc._neq_1rdm()

    #    for i,ref in enumerate(Aref):
    #        out = cc.compute_prop(field, 6*int(0.1/deltat) - 2)
    #        diff = abs(ref - out)
    #        #print((i + 1)*int(0.1/deltat))
    #        print("{} -- Expected: {}  Actual: {} ".format(i,ref,out))
    #        msg = "{} -- Expected: {}  Actual: {} ".format(i,ref,out)
    #        #self.assertTrue(diff < self.thresh, msg)


if __name__ == '__main__':
    unittest.main()
