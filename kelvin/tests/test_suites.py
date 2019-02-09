import sys
import unittest
from kelvin.tests import test_ccsd
from kelvin.tests import test_ft_cc_ampl
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
from kelvin.tests import test_scf
from kelvin.tests import test_test
from kelvin.tests import test_ueg

def full_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G"))
    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g_gen"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G_gen"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond_u"))

    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_ccsd_stanton"))
    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_uccsd"))

    suite.addTest(test_ft_ccsd.FTCCSDTest("test_2orb"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_rt"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_active"))
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

    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_ln"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_PUEG"))

    suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen"))
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

    suite.addTest(test_scf.SCFTest("test_Be_sto3g_energy"))
    suite.addTest(test_scf.SCFTest("test_diamond_energy"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_mp_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv"))

    suite.addTest(test_ueg.UEGTest("test_2_07_4"))
    suite.addTest(test_ueg.UEGTest("test_2_07_8"))

    return suite

def default_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))
 
    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_ccsd_stanton"))

    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G"))
    suite.addTest(test_ccsd.CCSDTest("test_Be_sto3g_gen"))
    suite.addTest(test_ccsd.CCSDTest("test_N2p_631G_gen"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond"))
    #suite.addTest(test_ccsd.CCSDTest("test_diamond_u"))

    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_ccsd_stanton"))
    suite.addTest(test_ft_cc_ampl.FTamplEquationsTest("test_uccsd"))

    suite.addTest(test_ft_ccsd.FTCCSDTest("test_2orb"))
    suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_rt"))
    #suite.addTest(test_ft_ccsd.FTCCSDTest("test_Be_active"))
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

    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_Be_sto3g_ln"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_UEG"))
    #suite.addTest(test_ft_deriv.FTDerivTest("test_UEG_gen"))
    suite.addTest(test_ft_deriv.FTDerivTest("test_PUEG"))

    #suite.addTest(test_ft_lambda.FTLambdaTest("test_Be_sto3g_gen"))
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

    suite.addTest(test_scf.SCFTest("test_Be_sto3g_energy"))
    suite.addTest(test_scf.SCFTest("test_diamond_energy"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_diamond_ft_deriv"))
    suite.addTest(test_scf.SCFTest("test_Be_sto3g_ft_mp_deriv"))
    #suite.addTest(test_scf.SCFTest("test_diamond_ft_mp_deriv"))

    suite.addTest(test_ueg.UEGTest("test_2_07_4"))
    #suite.addTest(test_ueg.UEGTest("test_2_07_8"))

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
