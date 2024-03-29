import unittest
import numpy
from pyscf import gto, scf
from cqcpy import integrals
from kelvin.h2_field_system import H2FieldSystem
from kelvin.td_ccsd import TDCCSD
from kelvin.kel_ccsd import KelCCSD


class KelCCSDTest(unittest.TestCase):
    def test_h2_field(self):
        beta = 0.6
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
        thresh = 1e-5

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
        A_ref = []

        for i in range(ngrid_ref):
            t = i*deltat
            ti[i] = t
            Ht = H + numpy.sin(omega*t)*Hint
            e, v = numpy.linalg.eigh(Ht)
            ee = numpy.exp(-deltat*1.j*e)
            U = numpy.einsum('ai,i,bi->ab', v, ee, numpy.conj(v))
            p = numpy.einsum('ps,pq,qr->sr', numpy.conj(U), p, U)
            if i % (ngrid_ref//10) == 0:
                A_ref.append(numpy.einsum('ij,ji->', p, Hint))

        A = []
        sys = H2FieldSystem(T, mu, omega)
        prop = {"tprop": "rk4", "lprop": "rk4"}
        mycc = TDCCSD(sys, prop, T=T, mu=mu, iprint=0,
                      ngrid=80, saveT=True, saveL=True)
        mycc.run()
        mycc._ccsd_lambda()

        kccsd = KelCCSD(sys, prop, T=T, mu=mu, iprint=0)
        kccsd.init_from_ftccsd(mycc, contour="keldysh")
        kccsd._ccsd(nstep=200, step=0.005)
        A = []
        for i, p in enumerate(kccsd.P):
            if i % 20 == 0:
                A.append(numpy.einsum('ij,ji->', field, p))

        for i, out in enumerate(A):
            ref = A_ref[i]
            diff = abs(ref - out)
            msg = "{} -- Expected: {}  Actual: {} ".format(i, ref, out)
            self.assertTrue(diff < thresh, msg)

    def test_h2_field_save(self):
        beta = 0.6
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

        E = numpy.zeros((3))
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq', mos, field, mos)

        sys = H2FieldSystem(T, mu, omega)
        prop = {"tprop": "rk1", "lprop": "rk1"}
        mycc = TDCCSD(sys, prop, T=T, mu=mu, iprint=0,
                      ngrid=80, saveT=True, saveL=True)
        mycc.run()
        mycc._ccsd_lambda()

        kccsd = KelCCSD(sys, prop, T=T, mu=mu, iprint=0)
        kccsd.init_from_ftccsd(mycc, contour="keldysh")
        kccsd._ccsd(nstep=200, step=0.005)
        Aref = []
        for i, p in enumerate(kccsd.P):
            if i % 20 == 0:
                Aref.append(numpy.einsum('ij,ji->', field, p))

        kccsd2 = KelCCSD(sys, prop, T=T, mu=mu, iprint=0)
        kccsd2.init_from_ftccsd(mycc, contour="keldysh")
        kccsd2._ccsd(nstep=200, step=0.005, save=20)
        A = []
        for i, p in enumerate(kccsd2.P):
            A.append(numpy.einsum('ij,ji->', field, p))

        thresh = 1e-12
        for i, out in enumerate(A):
            ref = Aref[i]
            diff = abs(ref - out)
            msg = "{} -- Expected: {}  Actual: {} ".format(i, ref, out)
            self.assertTrue(diff < thresh, msg)


if __name__ == '__main__':
    unittest.main()
