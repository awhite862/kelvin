import unittest
import numpy
from pyscf import gto, scf
from cqcpy import cc_energy
from cqcpy import cc_equations
from cqcpy import spin_utils
from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem


def test_L1(cc, thresh):
    eo,ev = cc.sys.g_energies()
    Dov = 1.0/(eo[:,None] - ev[None,:])
    Doovv = 1.0/(eo[:,None,None,None] + eo[None,:,None,None]
                 - ev[None,None,:,None] - ev[None,None,None,:])
    Nov = 1.0/Dov
    Noovv = 1.0/Doovv
    no = eo.shape[0]
    nv = ev.shape[0]

    # get Fock matrix
    F = cc.sys.g_fock()
    F.oo = F.oo - numpy.diag(eo)  # subtract diagonal
    F.vv = F.vv - numpy.diag(ev)  # subtract diagonal

    # get ERIs
    I = cc.sys.g_aint()
    # get amplitudes
    if cc.sys.has_u():
        eoa, eva, eob, evb = cc.sys.u_energies()
        noa = eoa.shape[0]
        nva = eva.shape[0]
        nob = eob.shape[0]
        nvb = evb.shape[0]
        T1 = spin_utils.T1_to_spin(cc.T1[0], cc.T1[1], noa, nva, nob, nvb)
        T2 = spin_utils.T2_to_spin(cc.T2[0], cc.T2[1], cc.T2[2], noa, nva, nob, nvb)
        L1 = spin_utils.T1_to_spin(cc.L1[0], cc.L1[1], nva, noa, nvb, nob)
        L2 = spin_utils.T2_to_spin(cc.L2[0], cc.L2[1], cc.L2[2], nva, noa, nvb, nob)
    else:
        T1 = cc.T1
        T2 = cc.T2
        L1 = cc.L1
        L2 = cc.L2

    delta = 1e-4
    for i in range(no):
        for a in range(nv):
            TF = T1.copy()
            TB = T1.copy()
            TF[a,i] += delta
            TB[a,i] -= delta
            EF = cc_energy.cc_energy(TF, T2, F.ov, I.oovv)
            EB = cc_energy.cc_energy(TB, T2, F.ov, I.oovv)
            TF1, TF2 = cc_equations.ccsd_stanton(F, I, TF, T2)
            TB1, TB2 = cc_equations.ccsd_stanton(F, I, TB, T2)
            TF2 -= numpy.einsum('abij,ijab->abij', T2, Noovv)
            TB2 -= numpy.einsum('abij,ijab->abij', T2, Noovv)
            TF1 -= numpy.einsum('ai,ia->ai', TF, Nov)
            TB1 -= numpy.einsum('ai,ia->ai', TB, Nov)
            TEf = 0.25*numpy.einsum('ijab,abij->', L2, TF2)
            TEb = 0.25*numpy.einsum('ijab,abij->', L2, TB2)
            TEf += numpy.einsum('ia,ai', L1, TF1)
            TEb += numpy.einsum('ia,ai', L1, TB1)
            fw = EF + TEf
            bw = EB + TEb
            diff = (fw - bw)/(2*delta)
            if numpy.abs(diff) > thresh:
                return('{} {}: {}'.format(i, a, diff), False)
    return ("pass",True)


def test_L2(cc, thresh):
    eo,ev = cc.sys.g_energies()
    Dov = 1.0/(eo[:,None] - ev[None,:])
    Doovv = 1.0/(eo[:,None,None,None] + eo[None,:,None,None]
                 - ev[None,None,:,None] - ev[None,None,None,:])
    Nov = 1.0/Dov
    Noovv = 1.0/Doovv
    no = eo.shape[0]
    nv = ev.shape[0]

    # get Fock matrix
    F = cc.sys.g_fock()
    F.oo = F.oo - numpy.diag(eo)  # subtract diagonal
    F.vv = F.vv - numpy.diag(ev)  # subtract diagonal

    # get ERIs
    I = cc.sys.g_aint()

    # get amplitudes
    if cc.sys.has_u():
        eoa, eva, eob, evb = cc.sys.u_energies()
        noa = eoa.shape[0]
        nva = eva.shape[0]
        nob = eob.shape[0]
        nvb = evb.shape[0]
        T1 = spin_utils.T1_to_spin(cc.T1[0], cc.T1[1], noa, nva, nob, nvb)
        T2 = spin_utils.T2_to_spin(cc.T2[0], cc.T2[1], cc.T2[2], noa, nva, nob, nvb)
        L1 = spin_utils.T1_to_spin(cc.L1[0], cc.L1[1], nva, noa, nvb, nob)
        L2 = spin_utils.T2_to_spin(cc.L2[0], cc.L2[1], cc.L2[2], nva, noa, nvb, nob)
    else:
        T1 = cc.T1
        T2 = cc.T2
        L1 = cc.L1
        L2 = cc.L2

    delta = 1e-4
    for i in range(no):
        for j in range(no):
            for a in range(nv):
                for b in range(nv):
                    TF = T2.copy()
                    TB = T2.copy()
                    TF[a,b,i,j] += delta
                    TF[b,a,i,j] -= delta
                    TF[b,a,j,i] += delta
                    TF[a,b,j,i] -= delta
                    TB[a,b,i,j] -= delta
                    TB[b,a,i,j] += delta
                    TB[a,b,j,i] += delta
                    TB[b,a,j,i] -= delta
                    EF = cc_energy.cc_energy(T1, TF, F.ov, I.oovv)
                    EB = cc_energy.cc_energy(T1, TB, F.ov, I.oovv)
                    if cc.singles:
                        TF1,TF2 = cc_equations.ccsd_stanton(F, I, T1, TF)
                        TB1,TB2 = cc_equations.ccsd_stanton(F, I, T1, TB)
                    else:
                        TF2 = cc_equations.ccd_simple(F, I, TF)
                        TB2 = cc_equations.ccd_simple(F, I, TB)
                    TF2 -= numpy.einsum('abij,ijab->abij', TF, Noovv)
                    TB2 -= numpy.einsum('abij,ijab->abij', TB, Noovv)
                    if cc.singles:
                        TF1 -= numpy.einsum('ai,ia->ai', T1, Nov)
                        TB1 -= numpy.einsum('ai,ia->ai', T1, Nov)
                    TEf = 0.25*numpy.einsum('ijab,abij->', L2, TF2)
                    TEb = 0.25*numpy.einsum('ijab,abij->', L2, TB2)
                    if cc.singles:
                        TEf += numpy.einsum('ia,ai', L1, TF1)
                        TEb += numpy.einsum('ia,ai', L1, TB1)
                    fw = EF + TEf
                    bw = EB + TEb
                    diff = (fw - bw)/(2*delta)
                    if abs(diff) > thresh:
                        return ('{} {} {} {}: {}'.format(i, j, a, b, diff), False)
    return ("pass",True)


class LambdaTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8

    # T1 is zero in this case, so only test T2
    def test_Be_sto3g_gen(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        sys = SCFSystem(m, 0.0, 0.0, orbtype='g')
        ccsd0 = ccsd(sys, iprint=0, max_iter=44, econv=1e-12)
        Etot,Ecc = ccsd0.run()
        ccsd0._ccsd_lambda()
        out = test_L2(ccsd0, self.thresh)
        self.assertTrue(out[1], out[0])

    def test_N2p_sto3g_gen(self):
        mol = gto.M(
            verbose=0,
            atom='N 0 0 0; N 0 0 1.1',
            basis='sto-3g',
            charge=1,
            spin=1)
        m = scf.UHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        sys = SCFSystem(m, 0.0, 0.0, orbtype='g')
        ccsd0 = ccsd(sys, iprint=0, max_iter=44, econv=1e-12, tconv=1e-10)
        Etot,Ecc = ccsd0.run()
        ccsd0._ccsd_lambda()
        outs = test_L1(ccsd0, self.thresh)
        self.assertTrue(outs[1], outs[0])
        outd = test_L2(ccsd0, self.thresh)
        self.assertTrue(outd[1], outd[0])

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        sys = SCFSystem(m, 0.0, 0.0)
        ccsd0 = ccsd(sys, iprint=0, max_iter=44, econv=1e-12)
        Etot,Ecc = ccsd0.run()
        ccsd0._uccsd_lambda()
        out = test_L2(ccsd0, self.thresh)
        self.assertTrue(out[1], out[0])

    def test_N2p_sto3g(self):
        mol = gto.M(
            verbose=0,
            atom='N 0 0 0; N 0 0 1.1',
            basis='sto-3g',
            charge=1,
            spin=1)
        m = scf.UHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        sys = SCFSystem(m, 0.0, 0.0)
        ccsd0 = ccsd(sys, iprint=0, max_iter=44, econv=1e-12)
        Etot,Ecc = ccsd0.run()
        ccsd0._uccsd_lambda()
        outs = test_L1(ccsd0, self.thresh)
        self.assertTrue(outs[1], outs[0])
        outd = test_L2(ccsd0, self.thresh)
        self.assertTrue(outd[1], outd[0])


if __name__ == '__main__':
    unittest.main()
