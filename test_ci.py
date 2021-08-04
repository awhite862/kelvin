import sys
import unittest
import logging
from kelvin.tests import test_cc_utils
from kelvin.tests import test_ccsd
from kelvin.tests import test_ft_cc_2rdm
from kelvin.tests import test_ft_cc_ampl
from kelvin.tests import test_ft_cc_relden
from kelvin.tests import test_ft_ccsd
from kelvin.tests import test_ft_ccsd_rdm
from kelvin.tests import test_ft_deriv
from kelvin.tests import test_ft_lambda
from kelvin.tests import test_ft_lambda_equations
from kelvin.tests import test_ft_mp2
from kelvin.tests import test_hubbard
from kelvin.tests import test_hubbard_field
from kelvin.tests import test_kel_ccsd
from kelvin.tests import test_lambda
from kelvin.tests import test_mp2
from kelvin.tests import test_neq_ccsd
from kelvin.tests import test_neq_density
from kelvin.tests import test_neq_lambda
from kelvin.tests import test_neq_lambda_equation
from kelvin.tests import test_neq_prop
from kelvin.tests import test_quadrature
from kelvin.tests import test_td_ccsd
from kelvin.tests import test_td_ccsd_ESN
from kelvin.tests import test_td_ccsd_lambda
from kelvin.tests import test_td_ccsd_1rdm
from kelvin.tests import test_td_ccsd_2rdm
from kelvin.tests import test_td_ccsd_relden
from kelvin.tests import test_scf
from kelvin.tests import test_ueg
from kelvin.tests import test_ueg_utils


def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_active"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_gen_deriv"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_deriv"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_active_deriv"))

    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G"))
    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g_gen"))

    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_hubbard"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be"))

    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_ccsd_stanton"))
    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_uccsd"))

    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_prop"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_hubbard"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Hubbard_gu"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_gen"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_scf_gen"))

    suite.addTest(test_ft_ccsd.FTCCSDTest("test_2orb"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_pueg"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_conv"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_rt"))

    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dia"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dba"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dji"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dai"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dcdab"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dciab"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dbcai"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dijab"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dbjai"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dabij"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_djkai"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dkaij"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_dklij"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_u1rdm"))
    suite.addTest(test_ft_ccsd_rdm.FTCCSD_RDMTest("test_u2rdm"))

    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_ln"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG2"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_PUEG"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_hubbard"))

    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_ccsd_opt"))
    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_uccsd_opt"))

    suite.addTest(test_ft_mp2.FTMP2Test("test_0T_Be_sto3g"))

    suite.addTest(test_hubbard.HubbardTest("test_2_1"))
    suite.addTest(test_hubbard.HubbardTest("test_4_1_pbc"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd_site"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd"))
    suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd"))

    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g"))
    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g_gen"))

    suite.addTest(test_mp2.MP2Test("test_N2p_631G"))

    suite.addTest(test_neq_density.NEQDensityTest("test_den"))

    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_lambda_opt"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L1_simple"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L2_simple"))

    suite.addTest(test_neq_prop.NEQPropTest("test_h2_field_deriv")) # 1s 1.4%
    suite.addTest(test_neq_prop.NEQPropTest("test_h2_field")) # 45s 1.4%
    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field2")) # 6m45s 1.4%

    suite.addTest(test_quadrature.QuadTest("test_int_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_Lint_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_ln"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_sin"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_exp"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_p"))

    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_r_vs_u_active"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_ccd"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk1"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk2"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_cn"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_am2"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_active")) # 45s 0.2%

    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen")) # 10w 0.4%
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG"))
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG_r_vs_u"))

    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4_omega"))

    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk1")) # 15s 0.1%
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk2")) # 21s 0.2%

    suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_r_vs_u"))

    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_r_vs_u"))
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_r_vs_u_active")) # 25s 0.2%

    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_deriv"))
    #suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_mp_deriv")) # 35s 0.1%
    suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv_k"))
    #suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv")) # 15s 0.3%

    suite.addTest(test_ueg.UEGTest("test_2_07_4"))

    suite.addTest(test_ueg_utils.UEGUtilsTest("test_integral_sym"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ft_uint"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ufock"))

    suite.addTest(test_hubbard_field.HubbardFieldTest("test_null_cc"))

    suite.addTest(test_kel_ccsd.KelCCSDTest("test_h2_field"))
    suite.addTest(test_kel_ccsd.KelCCSDTest("test_h2_field_save"))

    return suite


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.ERROR,
        stream=sys.stdout)
    runner = unittest.TextTestRunner()
    runner.run(get_suite())
