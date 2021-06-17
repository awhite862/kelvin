import unittest
import numpy
from pyscf import gto, scf
from cqcpy import ft_utils

from kelvin.scf_system import SCFSystem
from kelvin import cc_utils


class CCUtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def _test_fd(self, F, B, D, delta, name, thresh):
        fd = (F - B)/(2.0*delta)
        diff = numpy.linalg.norm(fd - D)
        self.assertTrue(diff < thresh, "Difference in " + name + ": {}".format(diff))

    def test_Be_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        T = 0.02
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-10
        athresh = 1e-40
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu+delta, orbtype='u')
        ea, eb = sys.u_energies_tot()
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        focca = [x for x in foa if x > athresh]
        fvira = [x for x in fva if x > athresh]
        foccb = [x for x in fob if x > athresh]
        fvirb = [x for x in fvb if x > athresh]
        iocca = [i for i, x in enumerate(foa) if x > athresh]
        ivira = [i for i, x in enumerate(fva) if x > athresh]
        ioccb = [i for i, x in enumerate(fob) if x > athresh]
        ivirb = [i for i, x in enumerate(fvb) if x > athresh]
        Fa, Fb, Ia, Ib, Iabab = cc_utils.uft_active_integrals(
                sys, ea, eb, focca, fvira, foccb, fvirb, iocca, ivira, ioccb, ivirb)
        Fga, Fgb, Iga, Igb, Igabab = cc_utils.uft_integrals(sys, ea, eb, beta, mu)

        # test Fock matrix
        Foo = Fga.oo[numpy.ix_(iocca, iocca)]
        Fov = Fga.oo[numpy.ix_(iocca, ivira)]
        Fvo = Fga.oo[numpy.ix_(ivira, iocca)]
        Fvv = Fga.oo[numpy.ix_(ivira, ivira)]
        doo = numpy.linalg.norm(Fa.oo - Foo)/numpy.sqrt(Foo.size)
        dov = numpy.linalg.norm(Fa.ov - Fov)/numpy.sqrt(Fov.size)
        dvo = numpy.linalg.norm(Fa.vo - Fvo)/numpy.sqrt(Fvo.size)
        dvv = numpy.linalg.norm(Fa.vv - Fvv)/numpy.sqrt(Fvv.size)
        self.assertTrue(doo < thresh, "Error in Fooa: {}".format(doo))
        self.assertTrue(doo < thresh, "Error in Fova: {}".format(dov))
        self.assertTrue(doo < thresh, "Error in Fvoa: {}".format(dvo))
        self.assertTrue(doo < thresh, "Error in Fvva: {}".format(dvv))

        Foo = Fgb.oo[numpy.ix_(ioccb, ioccb)]
        Fov = Fgb.oo[numpy.ix_(ioccb, ivirb)]
        Fvo = Fgb.oo[numpy.ix_(ivirb, ioccb)]
        Fvv = Fgb.oo[numpy.ix_(ivirb, ivirb)]
        doo = numpy.linalg.norm(Fa.oo - Foo)/numpy.sqrt(Foo.size)
        dov = numpy.linalg.norm(Fa.ov - Fov)/numpy.sqrt(Fov.size)
        dvo = numpy.linalg.norm(Fa.vo - Fvo)/numpy.sqrt(Fvo.size)
        dvv = numpy.linalg.norm(Fa.vv - Fvv)/numpy.sqrt(Fvv.size)
        self.assertTrue(doo < thresh, "Error in Foob: {}".format(doo))
        self.assertTrue(doo < thresh, "Error in Fovb: {}".format(dov))
        self.assertTrue(doo < thresh, "Error in Fvob: {}".format(dvo))
        self.assertTrue(doo < thresh, "Error in Fvvb: {}".format(dvv))

        # test ERIs
        Ivvvv = Iga.vvvv[numpy.ix_(ivira, ivira, ivira, ivira)]
        Ivvvo = Iga.vvvo[numpy.ix_(ivira, ivira, ivira, iocca)]
        Ivovv = Iga.vovv[numpy.ix_(ivira, iocca, ivira, ivira)]
        Ivvoo = Iga.vvoo[numpy.ix_(ivira, ivira, iocca, iocca)]
        Ioovv = Iga.oovv[numpy.ix_(iocca, iocca, ivira, ivira)]
        Ivovo = Iga.vovo[numpy.ix_(ivira, iocca, ivira, iocca)]
        Ivooo = Iga.vooo[numpy.ix_(ivira, iocca, iocca, iocca)]
        Iooov = Iga.ooov[numpy.ix_(iocca, iocca, iocca, ivira)]
        Ioooo = Iga.oooo[numpy.ix_(iocca, iocca, iocca, iocca)]
        Dvvvv = numpy.linalg.norm(Ia.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(Ia.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvovv = numpy.linalg.norm(Ia.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dvvoo = numpy.linalg.norm(Ia.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(Ia.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(Ia.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dvooo = numpy.linalg.norm(Ia.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dooov = numpy.linalg.norm(Ia.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(Ia.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvva: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvoa: {}".format(Dvvvo))
        self.assertTrue(Dvovv < thresh, "Error in Ivovva: {}".format(Dvovv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvooa: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovva: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovoa: {}".format(Dvovo))
        self.assertTrue(Dvooo < thresh, "Error in Ivoooa: {}".format(Dvooo))
        self.assertTrue(Dooov < thresh, "Error in Iooova: {}".format(Dooov))

        Ivvvv = Igb.vvvv[numpy.ix_(ivirb, ivirb, ivirb, ivirb)]
        Ivvvo = Igb.vvvo[numpy.ix_(ivirb, ivirb, ivirb, ioccb)]
        Ivovv = Igb.vovv[numpy.ix_(ivirb, ioccb, ivirb, ivirb)]
        Ivvoo = Igb.vvoo[numpy.ix_(ivirb, ivirb, ioccb, ioccb)]
        Ioovv = Igb.oovv[numpy.ix_(ioccb, ioccb, ivirb, ivirb)]
        Ivovo = Igb.vovo[numpy.ix_(ivirb, ioccb, ivirb, ioccb)]
        Ivooo = Igb.vooo[numpy.ix_(ivirb, ioccb, ioccb, ioccb)]
        Iooov = Igb.ooov[numpy.ix_(ioccb, ioccb, ioccb, ivirb)]
        Ioooo = Igb.oooo[numpy.ix_(ioccb, ioccb, ioccb, ioccb)]
        Dvvvv = numpy.linalg.norm(Ib.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(Ib.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvovv = numpy.linalg.norm(Ib.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dvvoo = numpy.linalg.norm(Ib.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(Ib.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(Ib.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dvooo = numpy.linalg.norm(Ib.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dooov = numpy.linalg.norm(Ib.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(Ib.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvvb: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvob: {}".format(Dvvvo))
        self.assertTrue(Dvovv < thresh, "Error in Ivovvb: {}".format(Dvovv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvoob: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovvb: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovob: {}".format(Dvovo))
        self.assertTrue(Dvooo < thresh, "Error in Ivooob: {}".format(Dvooo))
        self.assertTrue(Dooov < thresh, "Error in Iooovb: {}".format(Dooov))
        self.assertTrue(Doooo < thresh, "Error in Ioooob: {}".format(Doooo))

        Ivvvv = Igabab.vvvv[numpy.ix_(ivira, ivirb, ivira, ivirb)]
        Ivvvo = Igabab.vvvo[numpy.ix_(ivira, ivirb, ivira, ioccb)]
        Ivvov = Igabab.vvov[numpy.ix_(ivira, ivirb, iocca, ivirb)]
        Ivovv = Igabab.vovv[numpy.ix_(ivira, ioccb, ivira, ivirb)]
        Iovvv = Igabab.ovvv[numpy.ix_(iocca, ivirb, ivira, ivirb)]
        Ivvoo = Igabab.vvoo[numpy.ix_(ivira, ivirb, iocca, ioccb)]
        Ioovv = Igabab.oovv[numpy.ix_(iocca, ioccb, ivira, ivirb)]
        Ivovo = Igabab.vovo[numpy.ix_(ivira, ioccb, ivira, ioccb)]
        Ivoov = Igabab.voov[numpy.ix_(ivira, ioccb, iocca, ivirb)]
        Iovvo = Igabab.ovvo[numpy.ix_(iocca, ivirb, ivira, ioccb)]
        Iovov = Igabab.ovov[numpy.ix_(iocca, ivirb, iocca, ivirb)]
        Ivooo = Igabab.vooo[numpy.ix_(ivira, ioccb, iocca, ioccb)]
        Iovoo = Igabab.ovoo[numpy.ix_(iocca, ivirb, iocca, ioccb)]
        Ioovo = Igabab.oovo[numpy.ix_(iocca, ioccb, ivira, ioccb)]
        Iooov = Igabab.ooov[numpy.ix_(iocca, ioccb, iocca, ivirb)]
        Ioooo = Igabab.oooo[numpy.ix_(iocca, ioccb, iocca, ioccb)]
        Dvvvv = numpy.linalg.norm(Iabab.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(Iabab.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvvov = numpy.linalg.norm(Iabab.vvov - Ivvov)/numpy.sqrt(Ivvov.size)
        Dvovv = numpy.linalg.norm(Iabab.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dovvv = numpy.linalg.norm(Iabab.ovvv - Iovvv)/numpy.sqrt(Iovvv.size)
        Dvvoo = numpy.linalg.norm(Iabab.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(Iabab.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(Iabab.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dovvo = numpy.linalg.norm(Iabab.ovvo - Iovvo)/numpy.sqrt(Iovvo.size)
        Dvoov = numpy.linalg.norm(Iabab.voov - Ivoov)/numpy.sqrt(Ivoov.size)
        Dovov = numpy.linalg.norm(Iabab.ovov - Iovov)/numpy.sqrt(Iovov.size)
        Dvooo = numpy.linalg.norm(Iabab.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dovoo = numpy.linalg.norm(Iabab.ovoo - Iovoo)/numpy.sqrt(Iovoo.size)
        Doovo = numpy.linalg.norm(Iabab.oovo - Ioovo)/numpy.sqrt(Ioovo.size)
        Dooov = numpy.linalg.norm(Iabab.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(Iabab.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvvab: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvoab: {}".format(Dvvvo))
        self.assertTrue(Dvvov < thresh, "Error in Ivvovab: {}".format(Dvvov))
        self.assertTrue(Dvovv < thresh, "Error in Ivovvab: {}".format(Dvovv))
        self.assertTrue(Dovvv < thresh, "Error in Iovvvab: {}".format(Dovvv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvooab: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovvab: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovoab: {}".format(Dvovo))
        self.assertTrue(Dovvo < thresh, "Error in Iovvoab: {}".format(Dovvo))
        self.assertTrue(Dvoov < thresh, "Error in Ivoovab: {}".format(Dvoov))
        self.assertTrue(Dovov < thresh, "Error in Iovovab: {}".format(Dovov))
        self.assertTrue(Dvooo < thresh, "Error in Ivoooab: {}".format(Dvooo))
        self.assertTrue(Dovoo < thresh, "Error in Iovooab: {}".format(Dovoo))
        self.assertTrue(Doovo < thresh, "Error in Iooovab: {}".format(Doovo))
        self.assertTrue(Dooov < thresh, "Error in Iooovab: {}".format(Dooov))
        self.assertTrue(Doooo < thresh, "Error in Iooooab: {}".format(Doooo))

    def test_Be_gen_deriv(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        T = 1.0
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-8
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu+delta, orbtype='g')
        en = sys.g_energies_tot()
        Ff, If = cc_utils.ft_integrals(sys, en, beta, mu + delta)
        sys = SCFSystem(m, T, mu-delta, orbtype='g')
        en = sys.g_energies_tot()
        Fb, Ib = cc_utils.ft_integrals(sys, en, beta, mu - delta)

        sys = SCFSystem(m, T, mu, orbtype='g')
        en = sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        dvec = -beta*numpy.ones(en.shape)
        dF, dI = cc_utils.ft_d_integrals(sys, en, fo, fv, dvec)

        self._test_fd(Ff.oo, Fb.oo, dF.oo, delta, "Foo", thresh)
        self._test_fd(Ff.ov, Fb.ov, dF.ov, delta, "Fov", thresh)
        self._test_fd(Ff.vo, Fb.vo, dF.vo, delta, "Fvo", thresh)
        self._test_fd(Ff.vv, Fb.vv, dF.vv, delta, "Fvv", thresh)

        self._test_fd(If.vvvv, Ib.vvvv, dI.vvvv, delta, "Ivvvv", thresh)
        self._test_fd(If.vvvo, Ib.vvvo, dI.vvvo, delta, "Ivvvo", thresh)
        self._test_fd(If.vovv, Ib.vovv, dI.vovv, delta, "Ivovv", thresh)
        self._test_fd(If.vvoo, Ib.vvoo, dI.vvoo, delta, "Ivvoo", thresh)
        self._test_fd(If.vovo, Ib.vovo, dI.vovo, delta, "Ivovo", thresh)
        self._test_fd(If.oovv, Ib.oovv, dI.oovv, delta, "Ioovv", thresh)
        self._test_fd(If.vooo, Ib.vooo, dI.vooo, delta, "Ivooo", thresh)
        self._test_fd(If.ooov, Ib.ooov, dI.ooov, delta, "Iooov", thresh)
        self._test_fd(If.oooo, Ib.oooo, dI.oooo, delta, "Ioooo", thresh)

    def test_Be_deriv(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        T = 1.0
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-8
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu+delta, orbtype='u')
        ea, eb = sys.u_energies_tot()
        Faf, Fbf, Iaf, Ibf, Iababf = cc_utils.uft_integrals(sys, ea, eb, beta, mu + delta)
        sys = SCFSystem(m, T, mu-delta, orbtype='u')
        ea, eb = sys.u_energies_tot()
        Fab, Fbb, Iab, Ibb, Iababb = cc_utils.uft_integrals(sys, ea, eb, beta, mu - delta)

        sys = SCFSystem(m, T, mu, orbtype='u')
        ea, eb = sys.u_energies_tot()
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        dveca = -beta*numpy.ones(ea.shape)
        dvecb = -beta*numpy.ones(eb.shape)
        dFa, dFb, dIa, dIb, dIabab = cc_utils.u_ft_d_integrals(
                sys, ea, eb, foa, fva, fob, fvb, dveca, dvecb)

        self._test_fd(Faf.oo, Fab.oo, dFa.oo, delta, "Faoo", thresh)
        self._test_fd(Faf.ov, Fab.ov, dFa.ov, delta, "Faov", thresh)
        self._test_fd(Faf.vo, Fab.vo, dFa.vo, delta, "Favo", thresh)
        self._test_fd(Faf.vv, Fab.vv, dFa.vv, delta, "Favv", thresh)

        self._test_fd(Fbf.oo, Fbb.oo, dFb.oo, delta, "Fboo", thresh)
        self._test_fd(Fbf.ov, Fbb.ov, dFb.ov, delta, "Fbov", thresh)
        self._test_fd(Fbf.vo, Fbb.vo, dFb.vo, delta, "Fbvo", thresh)
        self._test_fd(Fbf.vv, Fbb.vv, dFb.vv, delta, "Fbvv", thresh)

        self._test_fd(Iaf.vvvv, Iab.vvvv, dIa.vvvv, delta, "Iavvvv", thresh)
        self._test_fd(Iaf.vvvo, Iab.vvvo, dIa.vvvo, delta, "Iavvvo", thresh)
        self._test_fd(Iaf.vovv, Iab.vovv, dIa.vovv, delta, "Iavovv", thresh)
        self._test_fd(Iaf.vvoo, Iab.vvoo, dIa.vvoo, delta, "Iavvoo", thresh)
        self._test_fd(Iaf.vovo, Iab.vovo, dIa.vovo, delta, "Iavovo", thresh)
        self._test_fd(Iaf.oovv, Iab.oovv, dIa.oovv, delta, "Iaoovv", thresh)
        self._test_fd(Iaf.vooo, Iab.vooo, dIa.vooo, delta, "Iavooo", thresh)
        self._test_fd(Iaf.ooov, Iab.ooov, dIa.ooov, delta, "Iaooov", thresh)
        self._test_fd(Iaf.oooo, Iab.oooo, dIa.oooo, delta, "Iaoooo", thresh)

        self._test_fd(Ibf.vvvv, Ibb.vvvv, dIb.vvvv, delta, "Ibvvvv", thresh)
        self._test_fd(Ibf.vvvo, Ibb.vvvo, dIb.vvvo, delta, "Ibvvvo", thresh)
        self._test_fd(Ibf.vovv, Ibb.vovv, dIb.vovv, delta, "Ibvovv", thresh)
        self._test_fd(Ibf.vvoo, Ibb.vvoo, dIb.vvoo, delta, "Ibvvoo", thresh)
        self._test_fd(Ibf.vovo, Ibb.vovo, dIb.vovo, delta, "Ibvovo", thresh)
        self._test_fd(Ibf.oovv, Ibb.oovv, dIb.oovv, delta, "Iboovv", thresh)
        self._test_fd(Ibf.vooo, Ibb.vooo, dIb.vooo, delta, "Ibvooo", thresh)
        self._test_fd(Ibf.ooov, Ibb.ooov, dIb.ooov, delta, "Ibooov", thresh)
        self._test_fd(Ibf.oooo, Ibb.oooo, dIb.oooo, delta, "Iboooo", thresh)

        self._test_fd(Iababf.vvvv, Iababb.vvvv, dIabab.vvvv, delta, "Iababvvvv", thresh)
        self._test_fd(Iababf.vvvo, Iababb.vvvo, dIabab.vvvo, delta, "Iababvvvo", thresh)
        self._test_fd(Iababf.vvov, Iababb.vvov, dIabab.vvov, delta, "Iababvvov", thresh)
        self._test_fd(Iababf.vovv, Iababb.vovv, dIabab.vovv, delta, "Iababvovv", thresh)
        self._test_fd(Iababf.ovvv, Iababb.ovvv, dIabab.ovvv, delta, "Iababovvv", thresh)
        self._test_fd(Iababf.vvoo, Iababb.vvoo, dIabab.vvoo, delta, "Iababvvoo", thresh)
        self._test_fd(Iababf.vovo, Iababb.vovo, dIabab.vovo, delta, "Iababvovo", thresh)
        self._test_fd(Iababf.ovvo, Iababb.ovvo, dIabab.ovvo, delta, "Iababovvo", thresh)
        self._test_fd(Iababf.voov, Iababb.voov, dIabab.voov, delta, "Iababvoov", thresh)
        self._test_fd(Iababf.ovov, Iababb.ovov, dIabab.ovov, delta, "Iababovov", thresh)
        self._test_fd(Iababf.oovv, Iababb.oovv, dIabab.oovv, delta, "Iababoovv", thresh)
        self._test_fd(Iababf.vooo, Iababb.vooo, dIabab.vooo, delta, "Iababvooo", thresh)
        self._test_fd(Iababf.ovoo, Iababb.ovoo, dIabab.ovoo, delta, "Iababovoo", thresh)
        self._test_fd(Iababf.oovo, Iababb.oovo, dIabab.oovo, delta, "Iababoovo", thresh)
        self._test_fd(Iababf.ooov, Iababb.ooov, dIabab.ooov, delta, "Iababooov", thresh)
        self._test_fd(Iababf.oooo, Iababb.oooo, dIabab.oooo, delta, "Iababoooo", thresh)

    def test_Be_gen_active_deriv(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        T = 0.02
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-10
        athresh = 1e-40
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu+delta, orbtype='g')
        en = sys.g_energies_tot()
        dvec = -beta*numpy.ones(en.shape)
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        iocc = [i for i, x in enumerate(fo) if x > athresh]
        ivir = [i for i, x in enumerate(fv) if x > athresh]
        F, I = cc_utils.ft_d_active_integrals(sys, en, fo, fv, iocc, ivir, dvec)
        Fg, Ig = cc_utils.ft_d_integrals(sys, en, fo, fv, dvec)

        # test Fock matrix
        Foo = Fg.oo[numpy.ix_(iocc, iocc)]
        Fov = Fg.oo[numpy.ix_(iocc, ivir)]
        Fvo = Fg.oo[numpy.ix_(ivir, iocc)]
        Fvv = Fg.oo[numpy.ix_(ivir, ivir)]
        doo = numpy.linalg.norm(F.oo - Foo)/numpy.sqrt(Foo.size)
        dov = numpy.linalg.norm(F.ov - Fov)/numpy.sqrt(Fov.size)
        dvo = numpy.linalg.norm(F.vo - Fvo)/numpy.sqrt(Fvo.size)
        dvv = numpy.linalg.norm(F.vv - Fvv)/numpy.sqrt(Fvv.size)
        self.assertTrue(doo < thresh, "Error in Foo: {}".format(doo))
        self.assertTrue(doo < thresh, "Error in Fov: {}".format(dov))
        self.assertTrue(doo < thresh, "Error in Fvo: {}".format(dvo))
        self.assertTrue(doo < thresh, "Error in Fvv: {}".format(dvv))

        # test ERIs
        Ivvvv = Ig.vvvv[numpy.ix_(ivir, ivir, ivir, ivir)]
        Ivvvo = Ig.vvvo[numpy.ix_(ivir, ivir, ivir, iocc)]
        Ivovv = Ig.vovv[numpy.ix_(ivir, iocc, ivir, ivir)]
        Ivvoo = Ig.vvoo[numpy.ix_(ivir, ivir, iocc, iocc)]
        Ioovv = Ig.oovv[numpy.ix_(iocc, iocc, ivir, ivir)]
        Ivovo = Ig.vovo[numpy.ix_(ivir, iocc, ivir, iocc)]
        Ivooo = Ig.vooo[numpy.ix_(ivir, iocc, iocc, iocc)]
        Iooov = Ig.ooov[numpy.ix_(iocc, iocc, iocc, ivir)]
        Ioooo = Ig.oooo[numpy.ix_(iocc, iocc, iocc, iocc)]
        Dvvvv = numpy.linalg.norm(I.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(I.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvovv = numpy.linalg.norm(I.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dvvoo = numpy.linalg.norm(I.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(I.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(I.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dvooo = numpy.linalg.norm(I.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dooov = numpy.linalg.norm(I.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(I.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvv: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvo: {}".format(Dvvvo))
        self.assertTrue(Dvovv < thresh, "Error in Ivovv: {}".format(Dvovv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvoo: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovv: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovo: {}".format(Dvovo))
        self.assertTrue(Dvooo < thresh, "Error in Ivooo: {}".format(Dvooo))
        self.assertTrue(Dooov < thresh, "Error in Iooov: {}".format(Dooov))
        self.assertTrue(Doooo < thresh, "Error in Ioooo: {}".format(Doooo))

    def test_Be_active_deriv(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        T = 0.05
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-10
        athresh = 1e-40
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu+delta, orbtype='u')
        ea, eb = sys.u_energies_tot()
        dveca = -beta*numpy.ones(ea.shape)
        dvecb = -beta*numpy.ones(eb.shape)
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        focca = [x for x in foa if x > athresh]
        fvira = [x for x in fva if x > athresh]
        foccb = [x for x in fob if x > athresh]
        fvirb = [x for x in fvb if x > athresh]
        iocca = [i for i, x in enumerate(foa) if x > athresh]
        ivira = [i for i, x in enumerate(fva) if x > athresh]
        ioccb = [i for i, x in enumerate(fob) if x > athresh]
        ivirb = [i for i, x in enumerate(fvb) if x > athresh]
        Fa, Fb, Ia, Ib, Iabab = cc_utils.uft_d_active_integrals(
                sys, ea, eb, focca, fvira, foccb, fvirb, iocca,
                ivira, ioccb, ivirb, dveca, dvecb)
        Fga, Fgb, Iga, Igb, Igabab = cc_utils.u_ft_d_integrals(
                sys, ea, eb, foa, fva, fob, fvb, dveca, dvecb)

        # test Fock matrix
        Foo = Fga.oo[numpy.ix_(iocca, iocca)]
        Fov = Fga.oo[numpy.ix_(iocca, ivira)]
        Fvo = Fga.oo[numpy.ix_(ivira, iocca)]
        Fvv = Fga.oo[numpy.ix_(ivira, ivira)]
        doo = numpy.linalg.norm(Fa.oo - Foo)/numpy.sqrt(Foo.size)
        dov = numpy.linalg.norm(Fa.ov - Fov)/numpy.sqrt(Fov.size)
        dvo = numpy.linalg.norm(Fa.vo - Fvo)/numpy.sqrt(Fvo.size)
        dvv = numpy.linalg.norm(Fa.vv - Fvv)/numpy.sqrt(Fvv.size)
        self.assertTrue(doo < thresh, "Error in Fooa: {}".format(doo))
        self.assertTrue(doo < thresh, "Error in Fova: {}".format(dov))
        self.assertTrue(doo < thresh, "Error in Fvoa: {}".format(dvo))
        self.assertTrue(doo < thresh, "Error in Fvva: {}".format(dvv))

        Foo = Fgb.oo[numpy.ix_(ioccb, ioccb)]
        Fov = Fgb.oo[numpy.ix_(ioccb, ivirb)]
        Fvo = Fgb.oo[numpy.ix_(ivirb, ioccb)]
        Fvv = Fgb.oo[numpy.ix_(ivirb, ivirb)]
        doo = numpy.linalg.norm(Fa.oo - Foo)/numpy.sqrt(Foo.size)
        dov = numpy.linalg.norm(Fa.ov - Fov)/numpy.sqrt(Fov.size)
        dvo = numpy.linalg.norm(Fa.vo - Fvo)/numpy.sqrt(Fvo.size)
        dvv = numpy.linalg.norm(Fa.vv - Fvv)/numpy.sqrt(Fvv.size)
        self.assertTrue(doo < thresh, "Error in Foob: {}".format(doo))
        self.assertTrue(doo < thresh, "Error in Fovb: {}".format(dov))
        self.assertTrue(doo < thresh, "Error in Fvob: {}".format(dvo))
        self.assertTrue(doo < thresh, "Error in Fvvb: {}".format(dvv))

        # test integrals
        Ivvvv = Iga.vvvv[numpy.ix_(ivira, ivira, ivira, ivira)]
        Ivvvo = Iga.vvvo[numpy.ix_(ivira, ivira, ivira, iocca)]
        Ivovv = Iga.vovv[numpy.ix_(ivira, iocca, ivira, ivira)]
        Ivvoo = Iga.vvoo[numpy.ix_(ivira, ivira, iocca, iocca)]
        Ioovv = Iga.oovv[numpy.ix_(iocca, iocca, ivira, ivira)]
        Ivovo = Iga.vovo[numpy.ix_(ivira, iocca, ivira, iocca)]
        Ivooo = Iga.vooo[numpy.ix_(ivira, iocca, iocca, iocca)]
        Iooov = Iga.ooov[numpy.ix_(iocca, iocca, iocca, ivira)]
        Ioooo = Iga.oooo[numpy.ix_(iocca, iocca, iocca, iocca)]
        Dvvvv = numpy.linalg.norm(Ia.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(Ia.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvovv = numpy.linalg.norm(Ia.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dvvoo = numpy.linalg.norm(Ia.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(Ia.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(Ia.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dvooo = numpy.linalg.norm(Ia.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dooov = numpy.linalg.norm(Ia.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(Ia.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvva: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvoa: {}".format(Dvvvo))
        self.assertTrue(Dvovv < thresh, "Error in Ivovva: {}".format(Dvovv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvooa: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovva: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovoa: {}".format(Dvovo))
        self.assertTrue(Dvooo < thresh, "Error in Ivoooa: {}".format(Dvooo))
        self.assertTrue(Dooov < thresh, "Error in Iooova: {}".format(Dooov))

        Ivvvv = Igb.vvvv[numpy.ix_(ivirb, ivirb, ivirb, ivirb)]
        Ivvvo = Igb.vvvo[numpy.ix_(ivirb, ivirb, ivirb, ioccb)]
        Ivovv = Igb.vovv[numpy.ix_(ivirb, ioccb, ivirb, ivirb)]
        Ivvoo = Igb.vvoo[numpy.ix_(ivirb, ivirb, ioccb, ioccb)]
        Ioovv = Igb.oovv[numpy.ix_(ioccb, ioccb, ivirb, ivirb)]
        Ivovo = Igb.vovo[numpy.ix_(ivirb, ioccb, ivirb, ioccb)]
        Ivooo = Igb.vooo[numpy.ix_(ivirb, ioccb, ioccb, ioccb)]
        Iooov = Igb.ooov[numpy.ix_(ioccb, ioccb, ioccb, ivirb)]
        Ioooo = Igb.oooo[numpy.ix_(ioccb, ioccb, ioccb, ioccb)]
        Dvvvv = numpy.linalg.norm(Ib.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(Ib.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvovv = numpy.linalg.norm(Ib.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dvvoo = numpy.linalg.norm(Ib.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(Ib.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(Ib.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dvooo = numpy.linalg.norm(Ib.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dooov = numpy.linalg.norm(Ib.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(Ib.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvvb: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvob: {}".format(Dvvvo))
        self.assertTrue(Dvovv < thresh, "Error in Ivovvb: {}".format(Dvovv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvoob: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovvb: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovob: {}".format(Dvovo))
        self.assertTrue(Dvooo < thresh, "Error in Ivooob: {}".format(Dvooo))
        self.assertTrue(Dooov < thresh, "Error in Iooovb: {}".format(Dooov))
        self.assertTrue(Doooo < thresh, "Error in Ioooob: {}".format(Doooo))

        Ivvvv = Igabab.vvvv[numpy.ix_(ivira, ivirb, ivira, ivirb)]
        Ivvvo = Igabab.vvvo[numpy.ix_(ivira, ivirb, ivira, ioccb)]
        Ivvov = Igabab.vvov[numpy.ix_(ivira, ivirb, iocca, ivirb)]
        Ivovv = Igabab.vovv[numpy.ix_(ivira, ioccb, ivira, ivirb)]
        Iovvv = Igabab.ovvv[numpy.ix_(iocca, ivirb, ivira, ivirb)]
        Ivvoo = Igabab.vvoo[numpy.ix_(ivira, ivirb, iocca, ioccb)]
        Ioovv = Igabab.oovv[numpy.ix_(iocca, ioccb, ivira, ivirb)]
        Ivovo = Igabab.vovo[numpy.ix_(ivira, ioccb, ivira, ioccb)]
        Ivoov = Igabab.voov[numpy.ix_(ivira, ioccb, iocca, ivirb)]
        Iovvo = Igabab.ovvo[numpy.ix_(iocca, ivirb, ivira, ioccb)]
        Iovov = Igabab.ovov[numpy.ix_(iocca, ivirb, iocca, ivirb)]
        Ivooo = Igabab.vooo[numpy.ix_(ivira, ioccb, iocca, ioccb)]
        Iovoo = Igabab.ovoo[numpy.ix_(iocca, ivirb, iocca, ioccb)]
        Ioovo = Igabab.oovo[numpy.ix_(iocca, ioccb, ivira, ioccb)]
        Iooov = Igabab.ooov[numpy.ix_(iocca, ioccb, iocca, ivirb)]
        Ioooo = Igabab.oooo[numpy.ix_(iocca, ioccb, iocca, ioccb)]
        Dvvvv = numpy.linalg.norm(Iabab.vvvv - Ivvvv)/numpy.sqrt(Ivvvv.size)
        Dvvvo = numpy.linalg.norm(Iabab.vvvo - Ivvvo)/numpy.sqrt(Ivvvo.size)
        Dvvov = numpy.linalg.norm(Iabab.vvov - Ivvov)/numpy.sqrt(Ivvov.size)
        Dvovv = numpy.linalg.norm(Iabab.vovv - Ivovv)/numpy.sqrt(Ivovv.size)
        Dovvv = numpy.linalg.norm(Iabab.ovvv - Iovvv)/numpy.sqrt(Iovvv.size)
        Dvvoo = numpy.linalg.norm(Iabab.vvoo - Ivvoo)/numpy.sqrt(Ivvoo.size)
        Doovv = numpy.linalg.norm(Iabab.oovv - Ioovv)/numpy.sqrt(Ioovv.size)
        Dvovo = numpy.linalg.norm(Iabab.vovo - Ivovo)/numpy.sqrt(Ivovo.size)
        Dovvo = numpy.linalg.norm(Iabab.ovvo - Iovvo)/numpy.sqrt(Iovvo.size)
        Dvoov = numpy.linalg.norm(Iabab.voov - Ivoov)/numpy.sqrt(Ivoov.size)
        Dovov = numpy.linalg.norm(Iabab.ovov - Iovov)/numpy.sqrt(Iovov.size)
        Dvooo = numpy.linalg.norm(Iabab.vooo - Ivooo)/numpy.sqrt(Ivooo.size)
        Dovoo = numpy.linalg.norm(Iabab.ovoo - Iovoo)/numpy.sqrt(Iovoo.size)
        Doovo = numpy.linalg.norm(Iabab.oovo - Ioovo)/numpy.sqrt(Ioovo.size)
        Dooov = numpy.linalg.norm(Iabab.ooov - Iooov)/numpy.sqrt(Iooov.size)
        Doooo = numpy.linalg.norm(Iabab.oooo - Ioooo)/numpy.sqrt(Ioooo.size)
        self.assertTrue(Dvvvv < thresh, "Error in Ivvvvab: {}".format(Dvvvv))
        self.assertTrue(Dvvvo < thresh, "Error in Ivvvoab: {}".format(Dvvvo))
        self.assertTrue(Dvvov < thresh, "Error in Ivvovab: {}".format(Dvvov))
        self.assertTrue(Dvovv < thresh, "Error in Ivovvab: {}".format(Dvovv))
        self.assertTrue(Dovvv < thresh, "Error in Iovvvab: {}".format(Dovvv))
        self.assertTrue(Dvvoo < thresh, "Error in Ivvooab: {}".format(Dvvoo))
        self.assertTrue(Doovv < thresh, "Error in Ioovvab: {}".format(Doovv))
        self.assertTrue(Dvovo < thresh, "Error in Ivovoab: {}".format(Dvovo))
        self.assertTrue(Dovvo < thresh, "Error in Iovvoab: {}".format(Dovvo))
        self.assertTrue(Dvoov < thresh, "Error in Ivoovab: {}".format(Dvoov))
        self.assertTrue(Dovov < thresh, "Error in Iovovab: {}".format(Dovov))
        self.assertTrue(Dvooo < thresh, "Error in Ivoooab: {}".format(Dvooo))
        self.assertTrue(Dovoo < thresh, "Error in Iovooab: {}".format(Dovoo))
        self.assertTrue(Doovo < thresh, "Error in Ioovoab: {}".format(Doovo))
        self.assertTrue(Dooov < thresh, "Error in Iooovab: {}".format(Dooov))
        self.assertTrue(Doooo < thresh, "Error in Iooooab: {}".format(Doooo))


if __name__ == '__main__':
    unittest.main()
