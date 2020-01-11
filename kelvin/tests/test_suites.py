import sys
import unittest
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
from kelvin.tests import test_test
from kelvin.tests import test_ueg
from kelvin.tests import test_ueg_utils

def full_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_active"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_gen_deriv"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_deriv"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_active_deriv"))

    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G"))
    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g_gen"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G_gen"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond_u"))

    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_hubbard"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_gen"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_gen_active"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_active"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_active_full"))

    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_ccsd_stanton"))
    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_uccsd"))

    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_prop"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_active"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_prop_active"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_pueg"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_gen"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_scf_gen"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_active"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Beplus"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_hubbard"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Hubbard_gu"))

    suite.addTest(test_ft_ccsd.FTCCSDTest("test_2orb"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_rt"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_active"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_uactive"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_pueg"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen_conv"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_conv"))

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

    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen_active"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_active"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_ln"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG2"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_PUEG"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_hubbard"))

    suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen"))
    suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen_active"))
    suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g"))
    suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_deriv"))

    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_ccsd_opt"))
    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_uccsd_opt"))

    suite.addTest(test_ft_mp2.FTMP2Test("test_Be_vs_fd_all"))
    suite.addTest(test_ft_mp2.FTMP2Test("test_0T_Be_sto3g"))
    suite.addTest(test_ft_mp2.FTMP2Test("test_Hdiamond_vs_fd"))

    suite.addTest(test_hubbard.HubbardTest("test_2_1"))
    suite.addTest(test_hubbard.HubbardTest("test_4_1_closed"))
    suite.addTest(test_hubbard.HubbardTest("test_4_1_open"))
    suite.addTest(test_hubbard.HubbardTest("test_6_2_open"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd_site"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd"))
    suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd"))
    suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd_u_g"))

    suite.addTest(test_hubbard_field.HubbardFieldTest("test_null_cc"))
    suite.addTest(test_hubbard_field.HubbardFieldTest("test_cc_vs_fci"))

    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g"))
    suite.addTest(test_lambda.LambdaTest("test_N2p_sto3g"))
    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g_gen"))
    suite.addTest(test_lambda.LambdaTest("test_N2p_sto3g_gen"))

    suite.addTest(test_mp2.MP2Test("test_Be_sto3g"))
    suite.addTest(test_mp2.MP2Test("test_N2p_631G"))

    suite.addTest(test_neq_ccsd.NEQ_CCSDTest("test_h2_field"))
    suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_null_field"))
    suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_field"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L1_simple"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L2_simple"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_lambda_opt"))
    suite.addTest(test_neq_density.NEQDensityTest("test_den"))
    suite.addTest(test_neq_prop.NEQPropTest("test_h2_field_deriv"))
    suite.addTest(test_neq_prop.NEQPropTest("test_h2_field"))
    suite.addTest(test_neq_prop.NEQPropTest("test_h2_field2"))

    suite.addTest(test_quadrature.QuadTest("test_int_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_Lint_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_ln"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_sin"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_exp"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_p"))

    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk1"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk2"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_cn"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_am2"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_active"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_u_vs_g"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_u_vs_g_active"))

    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen"))
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG_gen"))
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen_active"))
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG"))
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_active"))

    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4_omega"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_omega_active"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk1"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_cn"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk124"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_u_vs_g"))

    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk1"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk2"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_active"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_u_vs_g"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_u_vs_g_active"))

    suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_rk4_active"))
    suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_u_vs_g"))

    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4_active"))
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_u_vs_g"))

    suite.addTest(test_scf.SCFTest("test_Be_sto3g_energy"))
    suite.addTest(test_scf.SCFTest("test_diamond_energy"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_mp_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv"))

    suite.addTest(test_ueg.UEGTest("test_2_07_4"))
    suite.addTest(test_ueg.UEGTest("test_2_07_8"))

    suite.addTest(test_ueg_utils.UEGUtilsTest("test_integral_sym"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ft_uint"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ufock"))

    return suite

def default_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))
 
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_active"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_gen_deriv"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_deriv"))
    suite.addTest(test_cc_utils.CCUtilsTest("test_Be_active_deriv"))

    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G"))
    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g_gen"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G_gen"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond_u"))

    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_hubbard"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_gen"))
    #suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_gen_active"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be"))
    #suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_active"))
    suite.addTest(test_ft_cc_2rdm.FTCC2RDMTest("test_Be_active_full"))

    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_ccsd_stanton"))
    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_uccsd"))

    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_prop"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_active"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_gen_prop_active"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_pueg"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_gen"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg_scf_gen"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_ueg"))
    #suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Be_active"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Beplus"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_hubbard"))
    suite.addTest(test_ft_cc_relden.FTCCReldenTest("test_Hubbard_gu"))

    suite.addTest(test_ft_ccsd.FTCCSDTest("test_2orb"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_rt"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_active"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_uactive"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_pueg"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_gen_conv"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_ueg_conv"))

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

    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen_active"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_active"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_ln"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG2"))
    #suite.addTest(test_ft_deriv.FTDerivTest("test_UEG_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_PUEG"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_hubbard"))

    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen"))
    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen_active"))
    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g"))
    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_deriv"))

    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_ccsd_opt"))
    suite.addTest(test_ft_lambda_equations.FTLambdaEquationsTest("test_uccsd_opt"))

    suite.addTest(test_ft_mp2.FTMP2Test("test_Be_vs_fd_all"))
    #suite.addTest(test_ft_mp2.FTMP2Test("test_0T_Be_sto3g"))

    suite.addTest(test_hubbard.HubbardTest("test_2_1"))
    suite.addTest(test_hubbard.HubbardTest("test_4_1_closed"))
    suite.addTest(test_hubbard.HubbardTest("test_4_1_open"))
    #suite.addTest(test_hubbard.HubbardTest("test_6_2_open"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd_site"))
    suite.addTest(test_hubbard.HubbardTest("test_ccsd"))
    suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd"))
    #suite.addTest(test_hubbard.HubbardTest("test_ft_ccsd_u_g"))

    suite.addTest(test_hubbard_field.HubbardFieldTest("test_null_cc"))
    #suite.addTest(test_hubbard_field.HubbardFieldTest("test_cc_vs_fci"))

    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g"))
    #suite.addTest(test_lambda.LambdaTest("test_N2p_sto3g"))
    suite.addTest(test_lambda.LambdaTest("test_Be_sto3g_gen"))
    #suite.addTest(test_lambda.LambdaTest("test_N2p_sto3g_gen"))

    #suite.addTest(test_mp2.MP2Test("test_Be_sto3g"))
    suite.addTest(test_mp2.MP2Test("test_N2p_631G"))

    #suite.addTest(test_neq_ccsd.NEQ_CCSDTest("test_h2_field"))
    #suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_null_field"))
    #suite.addTest(test_neq_lambda.NEQLambdaTest("test_h2_field"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L1_simple"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_L2_simple"))
    suite.addTest(test_neq_lambda_equation.NEQLambdaEquationsTest("test_lambda_opt"))
    suite.addTest(test_neq_density.NEQDensityTest("test_den"))
    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field_deriv"))
    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field"))
    #suite.addTest(test_neq_prop.NEQPropTest("test_h2_field2"))

    suite.addTest(test_quadrature.QuadTest("test_int_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_Lint_keldysh"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_ln"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_sin"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_exp"))
    suite.addTest(test_quadrature.QuadTest("test_d_simpson_p"))

    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk1"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk2"))
    suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_rk4"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_cn"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_am2"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_active"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_u_vs_g"))
    #suite.addTest(test_td_ccsd.TDCCSDTest("test_Be_u_vs_g_active"))

    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen"))
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG_gen"))
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_gen_active"))
    suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_UEG"))
    #suite.addTest(test_td_ccsd_ESN.TDCCSDESNTest("test_Be_active"))

    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4_omega"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_omega_active"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk1"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_cn"))
    #suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_rk124"))
    suite.addTest(test_td_ccsd_lambda.TDCCSDLambdaTest("test_Be_u_vs_g"))

    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk1"))
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk2"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_active"))
    suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_u_vs_g"))
    #suite.addTest(test_td_ccsd_1rdm.TDCCSD1RDMTest("test_Be_u_vs_g_active"))

    #suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_rk4"))
    suite.addTest(test_td_ccsd_2rdm.TDCCSD2RDMTest("test_Be_rk4_active"))

    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4"))
    #suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_rk4_active"))
    suite.addTest(test_td_ccsd_relden.TDCCSDReldenTest("test_Be_u_vs_g"))

    suite.addTest(test_scf.SCFTest("test_Be_sto3g_energy"))
    suite.addTest(test_scf.SCFTest("test_diamond_energy"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_mp_deriv"))
    #suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv"))

    suite.addTest(test_ueg.UEGTest("test_2_07_4"))
    #suite.addTest(test_ueg.UEGTest("test_2_07_8"))

    suite.addTest(test_ueg_utils.UEGUtilsTest("test_integral_sym"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ft_uint"))
    suite.addTest(test_ueg_utils.UEGUtilsTest("test_ufock"))

    return suite

if __name__ == '__main__':
    if len(sys.argv) == 1:
        runner = unittest.TextTestRunner()
        runner.run(default_suite())
    elif sys.argv[1] == "full" or sys.argv[1] == "all":
        runner = unittest.TextTestRunner()
        runner.run(full_suite())
    else:
        raise Exception("Unrecognized argument")
