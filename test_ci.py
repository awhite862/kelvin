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
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Beplus")) # 55s 0%
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_pueg")) # 1m10s 0%
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_hubbard"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Hubbard_gu")) # 4s 0%
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_active")) # 1m50s 1%
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_prop_active")) # 3m22s 1%
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_gen"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_scf_gen")) #  10s 0%
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg")) # 22s 0%
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_active")) # 3m45s 3%

    suite.addTest(test_ft_ccsd.FTCCSDTest("test_2orb"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be")) # 5s 0%
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg")) # 20s 0%
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_pueg")) # 1s 0%
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen_conv"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_conv"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen")) # 1s 0%
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_rt"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_active")) # 1m45s 0%
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_uactive")) # 45s 0%

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

    #suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen")) # 2s 0%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen_active")) # 1m54s 1.2%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_active")) # 2m6s 2.7%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g")) # 2s 0.0%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_ln")) # 4s 0.1%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_UEG")) # 14s 0.0%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_UEG2")) # 7s 0.6%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_PUEG")) # 2s 0.6%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_hubbard")) # 4s 0.0%
    #suite.addTest(test_ft_deriv.FTDerivTest("test_UEG_gen")) # 17s 0.0%

    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen")) # 2m25s 0.0%
    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen_active")) $ 6m 0.5%
    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g")) # 2m30s 0.0%
    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_deriv")) # 2m20s 0.0%

    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_ccsd_opt"))
    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_uccsd_opt"))

    #suite.addTest(test_ft_mp2.FTMP2Test("test_Be_vs_fd_all")) # 55s 0.5%
    #suite.addTest(test_ft_mp2.FTMP2Test("test_0T_Be_sto3g")) # 1s 0.5%

    suite.addTest(test_hubbard.HubbardTest("test_2_1"))
    #suite.addTest(test_hubbard.HubbardTest("test_4_1"))
    suite.addTest(test_hubbard.HubbardTest("test_4_1_pbc"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd_site"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd"))
    #suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd")) # 5s 0.1%
    #suite.addTest(test_hubbard.HubbardTest("test_6_2_pbc")) # 6s 0.0%
    #suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd_u_g")) # 0s 0.0%

    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g"))
    #suite.addTest(test_lambda.LambdaTest("test_Be_sto3g_gen")) # 1s 0.4%
    #suite.addTest(test_lambda.LambdaTest("test_N2p_sto3g")) # 1m30s 0.0%

    suite.addTest(test_mp2.MP2Test("test_N2p_631G"))
    #suite.addTest(test_mp2.MP2Test("test_Be_sto3g")) # 1s 0.0%

    #suite.addTest(test_neq_ccsd.NEQ_CCSDTest("test_h2_field")) # 9m 0.6%
    #suite.addTest(test_neq_ccsd.NEQ_CCSDTest("test_h2_field2")) # 1m55s 1.7%
    #suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_null_field")) # 6m30s 1.1%
    #suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_field")) # 5m45s 1.1%

    suite.addTest(test_neq_density.NEQDensityTest("test_den"))

    #suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_null_field")) # 6m15s 1.1%
    #suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_field")) # 6m 1.1%

    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_lambda_opt"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L1_simple"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L2_simple"))

    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field_deriv")) # 1s 1.4%
    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field")) # 45s 1.4%
    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field2")) # 6m45s 1.4%

    suite.addTest(test_quadrature.QuadTest("test_int_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_Lint_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_ln"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_sin"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_exp"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_p"))

    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk4"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_r_vs_u_active")) # 1s 0.0%
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_ccd"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk1")) # 7s 0.1%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk2")) # 5s 0.1%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_cn")) # 6s 0.4%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_am2")) # 6s 0.4%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_active")) # 45s 0.2%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_u_vs_g")) # 5s 0.0%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_u_vs_g_active")) # 5s 0.0%
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_r_vs_u")) # 5s 0.0%

    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen")) # 10w 0.4%
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG"))
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG_r_vs_u")) # 16s 0.7%
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG_gen")) # 40s 0.5%
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen_active")) # 10m 1.6%
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG_h5py"))
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_active"))
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_active_r_vs_u")) # 45s 0.6%

    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4_omega"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_cn")) # 6m35s 0.6%
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_u_vs_g")) # 30s 0.1%
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_omega_active")) # 3s 0.0%
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk1")) # 45s 0.2%
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk124")) # 3m 0.3%
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_tsave"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_Lsave")) # 5s 0.0%
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_ccd")) # 5m30s 0.4%

    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk1")) # 15s 0.1%
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk4"))
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_active")) # 2m17s 0.6%
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_u_vs_g_active")) # 35s 0.1%
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_r_vs_u")) # 32s 0.1%
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk2")) # 21s 0.2%
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_u_vs_g")) # 31s 0.1%

    #suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_rk4_active")) # 6m40s 0.8%
    suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_r_vs_u"))
    #suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_rk4")) # 5m 0.1%
    #suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_u_vs_g")) # 1m10s 0.0%

    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_u_vs_g")) #  40s 0.0%
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_r_vs_u"))
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4"))
    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4_active")) # 2m40s 1.3%
    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_r_vs_u_active")) # 25s 0.2%
    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4_relden")) # 1m 0.4%
    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4_relden_r_vs_u")) # 35s 0.6%

    #suite.addTest(test_scf.SCFTest("test_Be_sto3g_energy")) # 12s 0.0%
    #suite.addTest(test_scf.SCFTest("test_diamond_energy")) # 15s 0.0%
    #suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_deriv")) # 0s 0.0%
    #suite.addTest(test_scf.SCFTest("test_diamond_ft_deriv")) # 2s 0.1%
    #suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_mp_deriv")) # 35s 0.1%
    #suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv_k")) # 11s 0.3%
    #suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv")) # 15s 0.3%

    suite.addTest(test_ueg.UEGTest("test_2_07_4"))
    #suite.addTest(test_ueg.UEGTest("test_2_07_8"))

    suite.addTest(test_ueg_utils.UEGUtilsTest("test_integral_sym"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ft_uint"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ufock"))

    suite.addTest(test_hubbard_field.HubbardFieldTest("test_null_cc"))
    #suite.addTest(test_hubbard_field.HubbardFieldTest("test_cc_vs_fci")) # 3m 0.8%

    suite.addTest(test_kel_ccsd.KelCCSDTest("test_h2_field"))
    #suite.addTest(test_kel_ccsd.KelCCSDTest("test_h2_field_save")) # 1s 0.2%

    return suite


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.ERROR,
        stream=sys.stdout)
    runner = unittest.TextTestRunner()
    runner.run(get_suite())
