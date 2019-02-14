import unittest
import numpy
from pyscf import gto, scf
from cqcpy import ft_utils

from kelvin.scf_system import scf_system
from kelvin import cc_utils

class CCUtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def _test_fd(self, F, B, D, delta, name, thresh):
        fd = (F - B)/(2.0*delta)
        diff = numpy.linalg.norm(fd - D) 
        self.assertTrue(diff < thresh,"Difference in " + name + ": {}".format(diff))

    def test_Be_gen(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        
        m = scf.RHF(mol)
        T = 1.0
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-8
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu+delta,orbtype='g')
        en = sys.g_energies_tot()
        Ff,If = cc_utils.get_ft_integrals(sys, en, beta, mu + delta)
        sys = scf_system(m,T,mu-delta,orbtype='g')
        en = sys.g_energies_tot()
        Fb,Ib = cc_utils.get_ft_integrals(sys, en, beta, mu - delta)

        sys = scf_system(m,T,mu,orbtype='g')
        en = sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        dvec = -beta*numpy.ones(en.shape)
        dF, dI = cc_utils.get_ft_d_integrals(sys, en, fo, fv, dvec)

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

    def test_Be(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        
        m = scf.RHF(mol)
        T = 1.0
        mu = 0.0
        beta = 1/T
        delta = 1e-4
        thresh = 1e-8
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu+delta,orbtype='u')
        ea,eb = sys.u_energies_tot()
        Faf,Fbf,Iaf,Ibf,Iababf = cc_utils.get_uft_integrals(sys, ea, eb, beta, mu + delta)
        sys = scf_system(m,T,mu-delta,orbtype='u')
        ea,eb = sys.u_energies_tot()
        Fab,Fbb,Iab,Ibb,Iababb = cc_utils.get_uft_integrals(sys, ea, eb, beta, mu - delta)

        sys = scf_system(m,T,mu,orbtype='u')
        ea,eb = sys.u_energies_tot()
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        dveca = -beta*numpy.ones(ea.shape)
        dvecb = -beta*numpy.ones(eb.shape)
        dFa,dFb,dIa,dIb,dIabab = cc_utils.u_ft_d_integrals(
                sys, ea, eb, foa, fob, fva, fvb, dveca, dvecb)

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

if __name__ == '__main__':
    unittest.main()
