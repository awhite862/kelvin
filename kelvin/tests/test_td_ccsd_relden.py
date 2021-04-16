import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

class TDCCSDReldenTest(unittest.TestCase):
    def test_Be_rk4(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 1.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()
        ccsdT._g_ft_2rdm()
        ccsdT._g_ft_ron()
        ccsdT._g_ft_rorb()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True, erel=True)
        tdccsdT._g_ft_ron()

        diff = numpy.linalg.norm(ccsdT.ron1 - tdccsdT.ron1)/numpy.linalg.norm(ccsdT.ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rono - tdccsdT.rono)/numpy.linalg.norm(ccsdT.rono)
        self.assertTrue(diff < 1e-5,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.ronv - tdccsdT.ronv)/numpy.linalg.norm(ccsdT.ronv)
        self.assertTrue(diff < 1e-5,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbo - tdccsdT.rorbo)/numpy.sqrt(float(ccsdT.rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbv - tdccsdT.rorbv)/numpy.sqrt(float(ccsdT.rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

    def test_Be_rk4_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.05
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=120,iprint=0,athresh=1e-20)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()
        ccsdT._g_ft_2rdm()
        ccsdT._g_ft_ron()
        ccsdT._g_ft_rorb()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=440, athresh=1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True,erel=True)
        tdccsdT._g_ft_ron()

        diff = numpy.linalg.norm(ccsdT.ron1 - tdccsdT.ron1)/numpy.linalg.norm(ccsdT.ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rono - tdccsdT.rono)/numpy.linalg.norm(ccsdT.rono)
        self.assertTrue(diff < 1e-2,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.ronv - tdccsdT.ronv)/numpy.linalg.norm(ccsdT.ronv)
        self.assertTrue(diff < 1e-2,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbo - tdccsdT.rorbo)/numpy.sqrt(float(ccsdT.rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbv - tdccsdT.rorbv)/numpy.sqrt(float(ccsdT.rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

    def test_Be_u_vs_g(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 1.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True, erel=True)
        tdccsdT._g_ft_ron()
        ron1 = tdccsdT.ron1
        rono = tdccsdT.rono
        ronv = tdccsdT.ronv
        rorbo = tdccsdT.rorbo
        rorbv = tdccsdT.rorbv

        sys = scf_system(m,T,mu,orbtype='u')
        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._uccsd_lambda(rdm2=True, erel=True)
        tdccsdT._u_ft_ron()
        uron1 = numpy.concatenate((tdccsdT.ron1[0], tdccsdT.ron1[1]))
        urono = numpy.concatenate((tdccsdT.rono[0], tdccsdT.rono[1]))
        uronv = numpy.concatenate((tdccsdT.ronv[0], tdccsdT.ronv[1]))
        urorbo = numpy.concatenate((tdccsdT.rorbo[0], tdccsdT.rorbo[1]))
        urorbv = numpy.concatenate((tdccsdT.rorbv[0], tdccsdT.rorbv[1]))

        diff = numpy.linalg.norm(ron1 - uron1)/numpy.linalg.norm(ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(rono - urono)/numpy.linalg.norm(rono)
        self.assertTrue(diff < 1e-5,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ronv - uronv)/numpy.linalg.norm(ronv)
        self.assertTrue(diff < 1e-5,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(rorbo - urorbo)/numpy.sqrt(float(rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(rorbv - urorbv)/numpy.sqrt(float(rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

    def test_Be_r_vs_u(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 1.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='r')

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._rccsd_lambda(rdm2=True, erel=True)
        tdccsdT._r_ft_ron()
        ron1 = tdccsdT.ron1
        rono = tdccsdT.rono
        ronv = tdccsdT.ronv
        rorbo = tdccsdT.rorbo
        rorbv = tdccsdT.rorbv

        sys = scf_system(m,T,mu,orbtype='u')
        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._uccsd_lambda(rdm2=True, erel=True)
        tdccsdT._u_ft_ron()
        uron1 = tdccsdT.ron1[0]
        urono = tdccsdT.rono[0]
        uronv = tdccsdT.ronv[0]
        urorbo = tdccsdT.rorbo[0]
        urorbv = tdccsdT.rorbv[0]

        diff = numpy.linalg.norm(ron1 - uron1)/numpy.linalg.norm(ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(rono - urono)/numpy.linalg.norm(rono)
        self.assertTrue(diff < 1e-5,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ronv - uronv)/numpy.linalg.norm(ronv)
        self.assertTrue(diff < 1e-5,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(rorbo - urorbo)/numpy.sqrt(float(rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(rorbv - urorbv)/numpy.sqrt(float(rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

    def test_Be_r_vs_u_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.05
        mu = 0.0
        athresh = 1e-20
        sys = scf_system(m,T,mu,orbtype='r')

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160, athresh=athresh, saveT=True)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._rccsd_lambda(rdm2=True, erel=True)
        tdccsdT._r_ft_ron()
        ron1 = tdccsdT.ron1
        rono = tdccsdT.rono
        ronv = tdccsdT.ronv
        rorbo = tdccsdT.rorbo
        rorbv = tdccsdT.rorbv

        sys = scf_system(m,T,mu,orbtype='u')
        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160, athresh=athresh, saveT=True)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._uccsd_lambda(rdm2=True, erel=True)
        tdccsdT._u_ft_ron()
        uron1 = tdccsdT.ron1[0]
        urono = tdccsdT.rono[0]
        uronv = tdccsdT.ronv[0]
        urorbo = tdccsdT.rorbo[0]
        urorbv = tdccsdT.rorbv[0]

        diff = numpy.linalg.norm(ron1 - uron1)/numpy.linalg.norm(ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(rono - urono)/numpy.linalg.norm(rono)
        self.assertTrue(diff < 1e-5,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ronv - uronv)/numpy.linalg.norm(ronv)
        self.assertTrue(diff < 1e-5,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(rorbo - urorbo)/numpy.sqrt(float(rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(rorbv - urorbv)/numpy.sqrt(float(rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

    def test_Be_rk4_relden(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 1.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='u')

        # compute normal-ordered 1-rdm
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._u_ft_1rdm()
        ccsdT._u_ft_2rdm()
        ccsdT._u_ft_ron()
        ccsdT._u_ft_rorb()
        rdm1_ref = ccsdT.full_1rdm(relax=True)

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._uccsd_lambda(rdm2=True, erel=True)
        rdm1_out = tdccsdT.full_1rdm(relax=True)
        d1 = numpy.linalg.norm(rdm1_ref[0] - rdm1_out[0])
        d2 = numpy.linalg.norm(rdm1_ref[1] - rdm1_out[1])

        self.assertTrue(d1 < 1e-6,"Error in alpha 1-rdm: {}".format(d1))
        self.assertTrue(d2 < 1e-6,"Error in alpha 1-rdm: {}".format(d2))

    def test_Be_rk4_relden_r_vs_u(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 1.0
        mu = 0.0

        # compute normal-ordered 1-rdm (U)
        sys = scf_system(m,T,mu,orbtype='u')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._uccsd_lambda(rdm2=True, erel=True)
        rdm1_ref = tdccsdT.full_1rdm(relax=True)

        # compute normal-order 1-rdm (R)
        sys = scf_system(m,T,mu,orbtype='r')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._rccsd_lambda(rdm2=True, erel=True)
        rdm1_out = tdccsdT.full_1rdm(relax=True)
        diff = numpy.linalg.norm(rdm1_ref[0] - rdm1_out)

        self.assertTrue(diff < 1e-12,"Error in 1-rdm: {}".format(diff))

if __name__ == '__main__':
    unittest.main()
