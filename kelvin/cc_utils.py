import time
import numpy
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from pyscf import lib
from . import ft_cc_energy
from . import ft_cc_equations
from . import quadrature

einsum = lib.einsum
#einsum = einsum

def form_new_ampl(method, F, I, T1old, T2old, D1, D2, ti, ng, G):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        ti (array): time grid.
        ng (int): number of time points.
        G (array): Quadrature weight matrix.
    """
    if method == "CCSD":
        T1,T2 = ft_cc_equations.ccsd_stanton(F,I,T1old,T2old,
                D1,D2,ti,ng,G)
    elif method == "CCD":
        T1 = T1old
        T2 = ft_cc_equations.ccd_simple(F,I,T2old,
                D2,ti,ng,G)
    elif method == "LCCSD":
        T1,T2 = ft_cc_equations.lccsd_simple(F,I,T1old,T2old,
                D1,D2,ti,ng,G)
    elif method == "LCCD":
        T1 = T1old
        T2 = ft_cc_equations.lccd_simple(F,I,T2old,
                D2,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword")

    return T1,T2

def form_new_ampl_u(method, Fa, Fb, Ia, Ib, Iabab, T1aold, T1bold, T2aaold, T2abold, T2bbold,
        D1a, D1b, D2aa, D2ab, D2bb, ti, ng, G):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        ti (array): time grid.
        ng (int): number of time points.
        G (array): Quadrature weight matrix.
    """
    if method == "CCSD":
        T1out,T2out = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)
    #elif method == "CCD":
    #    T1 = T1old
    #    T2 = ft_cc_equations.ccd_simple(F,I,T2old,
    #            D2,ti,ng,G)
    #elif method == "LCCSD":
    #    T1,T2 = ft_cc_equations.lccsd_simple(F,I,T1old,T2old,
    #            D1,D2,ti,ng,G)
    #elif method == "LCCD":
    #    T1 = T1old
    #    T2 = ft_cc_equations.lccd_simple(F,I,T2old,
    #            D2,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword for unrestricted calc")

    return T1out,T2out

def form_new_ampl_extrap(ig,method,F,I,T1,T2,T1bar,T2bar,D1,D2,ti,ng,G):
    if method == "CCSD":
        T1,T2 = ft_cc_equations.ccsd_stanton_single(ig,F,I,T1,T2,
                T1bar,T2bar,D1,D2,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword")
    return T1,T2

def form_new_ampl_extrap_u(ig,method,Fa,Fb,Ia,Ib,Iabab,
        T1a,T1b,T2aa,T2ab,T2bb,T1bara,T1barb,T2baraa,T2barab,T2barbb,
        D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G):
    if method == "CCSD":
        T1,T2 = ft_cc_equations.uccsd_stanton_single(ig,Fa,Fb,Ia,Ib,Iabab,
                T1a,T1b,T2aa,T2ab,T2bb,T1bara,T1barb,T2baraa,T2barab,T2barbb,
                D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword")
    return T1,T2

def ft_cc_iter(method, T1old, T2old, F, I, D1, D2, g, G, beta, ng, ti,
        iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    ethresh = conv_options["econv"]
    tthresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    i = 0
    Eold = 888888888.888888888
    nl1 = numpy.linalg.norm(T1old) + 0.1
    nl2 = numpy.linalg.norm(T2old) + 0.1
    while i < max_iter and not converged:
        # form new T1 and T2
        T1,T2 = form_new_ampl(method,F,I,T1old,T2old,D1,D2,ti,ng,G)

        res1 = numpy.linalg.norm(T1 - T1old) / nl1
        res2 = numpy.linalg.norm(T2 - T2old) / nl2
        # damp new T-amplitudes
        T1old = alpha*T1old + (1.0 - alpha)*T1
        T2old = alpha*T2old + (1.0 - alpha)*T2
        nl1 = numpy.linalg.norm(T1old) + 0.1
        nl2 = numpy.linalg.norm(T2old) + 0.1

        # compute energy
        E = ft_cc_energy.ft_cc_energy(T1old,T2old,
            F.ov,I.oovv,g,beta)

        # determine convergence
        if iprint > 0:
            if isinstance(E, complex):
                print(' %2d  %.10f  %.3E %.4E' % (i+1,E.real,E.imag,res1+res2))
            else:
                print(' %2d  %.10f   %.4E' % (i+1,E,res1+res2))
        i = i + 1
        if numpy.abs(E - Eold) < ethresh and res1+res2 < tthresh:
            converged = True
        Eold = E

    if not converged:
        print("WARNING: {} did not converge!".format(method))

    tend = time.time()
    if iprint > 0:
        print("Total {} time: {:.4f} s".format(method,(tend - tbeg)))

    return Eold,T1old,T2old

def ft_cc_iter_extrap(method, F, I, D1, D2, g, G, beta, ng, ti,
        iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]

    no,nv = F.ov.shape
    t1bar = numpy.zeros((ng,nv,no), dtype=F.vo.dtype)
    t2bar = numpy.zeros((ng,nv,nv,no,no), dtype=I.vvoo.dtype)
    T1new = numpy.zeros((ng,nv,no), dtype=t1bar.dtype)
    T2new = numpy.zeros((ng,nv,nv,no,no), dtype=t2bar.dtype)

    # loop over grid points
    for ig in range(ng):
        if ig == 0:
            t1bar[0] = -F.vo
            t2bar[0] = -I.vvoo
            continue # don't bother computing at T = inf
        elif ig == 1:
            t1bar[ig] = -F.vo
            t2bar[ig] = -I.vvoo
            T1new[ig] = quadrature.int_tbar1_single(ng,ig,t1bar,ti,D1,G)
            T2new[ig] = quadrature.int_tbar2_single(ng,ig,t2bar,ti,D2,G)
        else:
            # linear extrapolation
            T1new[ig] = T1new[ig - 1] + (T1new[ig - 2] - T1new[ig - 1])\
                    *(ti[ig] - ti[ig - 1])/(ti[ig - 2] - ti[ig - 1])
            T2new[ig] = T2new[ig - 1] + (T2new[ig - 2] - T2new[ig - 1])\
                    *(ti[ig] - ti[ig - 1])/(ti[ig - 2] - ti[ig - 1])
        converged = False
        nl1 = numpy.sqrt(float(T1new[ig].size))
        nl2 = numpy.sqrt(float(T2new[ig].size))
        if iprint > 0:
            print("Time point {}".format(ig))
        i = 0
        while i < max_iter and not converged:
            # form new T1 and T2
            T1,T2 = form_new_ampl_extrap(ig,method,F,I,T1new[ig],T2new[ig],
                    t1bar,t2bar,D1,D2,ti,ng,G)

            res1 = numpy.linalg.norm(T1 - T1new[ig]) / nl1
            res2 = numpy.linalg.norm(T2 - T2new[ig]) / nl2
            # damp new T-amplitudes
            T1new[ig] = alpha*T1new[ig] + (1.0 - alpha)*T1.copy()
            T2new[ig] = alpha*T2new[ig] + (1.0 - alpha)*T2.copy()

            # determine convergence
            if iprint > 0:
                print(' %2d  %.4E' % (i+1,res1+res2))
            i = i + 1
            if res1 + res2 < thresh:
                converged = True
    return T1new,T2new

def ft_ucc_iter(method, T1aold, T1bold, T2aaold, T2abold, T2bbold, Fa, Fb, Ia, Ib, Iabab,
        D1a, D1b, D2aa, D2ab, D2bb, g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    ethresh = conv_options["econv"]
    tthresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    i = 0
    na = D1a.shape[0]
    nb = D1b.shape[0]
    n = na + nb
    Eold = 888888888.888888888
    while i < max_iter and not converged:
        T1out,T2out = form_new_ampl_u(method,Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold, D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)

        nl1 = numpy.linalg.norm(T1aold) + 0.1
        nl1 += numpy.linalg.norm(T1bold)
        nl2 = numpy.linalg.norm(T2aaold) + 0.1
        nl2 += numpy.linalg.norm(T2abold)
        nl2 += numpy.linalg.norm(T2bbold)

        res1 = numpy.linalg.norm(T1out[0] - T1aold) / nl1
        res1 += numpy.linalg.norm(T1out[1] - T1bold) / nl1
        res2 = numpy.linalg.norm(T2out[0] - T2aaold) / nl2
        res2 += numpy.linalg.norm(T2out[1] - T2abold) / nl2
        res2 += numpy.linalg.norm(T2out[2] - T2bbold) / nl2

        # damp new T-amplitudes
        T1aold = alpha*T1aold + (1.0 - alpha)*T1out[0]
        T1bold = alpha*T1bold + (1.0 - alpha)*T1out[1]
        T2aaold = alpha*T2aaold + (1.0 - alpha)*T2out[0]
        T2abold = alpha*T2abold + (1.0 - alpha)*T2out[1]
        T2bbold = alpha*T2bbold + (1.0 - alpha)*T2out[2]

        # compute energy
        E = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,beta)

        # determine convergence
        if iprint > 0:
            if isinstance(E, complex):
                print(' %2d  %.10f  %.3E %.4E' % (i+1,E.real,E.imag,res1+res2))
            else:
                print(' %2d  %.10f   %.4E' % (i+1,E,res1+res2))
        i = i + 1
        if numpy.abs(E - Eold) < ethresh and res1+res2 < tthresh:
            converged = True
        Eold = E

    if not converged:
        print("WARNING: {} did not converge!".format(method))

    tend = time.time()
    if iprint > 0:
        print("Total {} time: {:.4f} s".format(method,(tend - tbeg)))

    return Eold,(T1aold,T1bold),(T2aaold,T2abold,T2bbold)

def ft_ucc_iter_extrap(method, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
        g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]

    noa,nva = Fa.ov.shape
    nob,nvb = Fb.ov.shape
    t1bara = numpy.zeros((ng,nva,noa), dtype=Fa.vo.dtype)
    t1barb = numpy.zeros((ng,nvb,nob), dtype=Fb.vo.dtype)
    t2baraa = numpy.zeros((ng,nva,nva,noa,noa), dtype=Ia.vvoo.dtype)
    t2barab = numpy.zeros((ng,nva,nvb,noa,nob), dtype=Iabab.vvoo.dtype)
    t2barbb = numpy.zeros((ng,nvb,nvb,nob,nob), dtype=Ib.vvoo.dtype)
    T1newa = numpy.zeros(t1bara.shape, dtype=t1bara.dtype)
    T1newb = numpy.zeros(t1barb.shape, dtype=t1barb.dtype)
    T2newaa = numpy.zeros(t2baraa.shape, dtype=t2baraa.dtype)
    T2newab = numpy.zeros(t2barab.shape, dtype=t2barab.dtype)
    T2newbb = numpy.zeros(t2barbb.shape, dtype=t2barbb.dtype)

    # loop over grid points
    for ig in range(ng):
        if ig == 0:
            t1bara[0] = -Fa.vo
            t1barb[0] = -Fb.vo
            t2baraa[0] = -Ia.vvoo
            t2barab[0] = -Iabab.vvoo
            t2barbb[0] = -Ib.vvoo
            continue # don't bother computing at T = inf
        elif ig == 1:
            t1bara[ig] = -Fa.vo
            t1barb[ig] = -Fb.vo
            t2baraa[ig] = -Ia.vvoo
            t2barab[ig] = -Iabab.vvoo
            t2barbb[ig] = -Ib.vvoo
            T1newa[ig] = quadrature.int_tbar1_single(ng,ig,t1bara,ti,D1a,G)
            T1newb[ig] = quadrature.int_tbar1_single(ng,ig,t1barb,ti,D1b,G)
            T2newaa[ig] = quadrature.int_tbar2_single(ng,ig,t2baraa,ti,D2aa,G)
            T2newab[ig] = quadrature.int_tbar2_single(ng,ig,t2barab,ti,D2ab,G)
            T2newbb[ig] = quadrature.int_tbar2_single(ng,ig,t2barbb,ti,D2bb,G)
        else:
            # linear extrapolation
            fac = (ti[ig] - ti[ig - 1])/(ti[ig - 2] - ti[ig - 1])
            T1newa[ig] = T1newa[ig - 1] + (T1newa[ig - 2] - T1newa[ig - 1])*fac
            T1newb[ig] = T1newb[ig - 1] + (T1newb[ig - 2] - T1newb[ig - 1])*fac
            T2newaa[ig] = T2newaa[ig - 1] + (T2newaa[ig - 2] - T2newaa[ig - 1])*fac
            T2newab[ig] = T2newab[ig - 1] + (T2newab[ig - 2] - T2newab[ig - 1])*fac
            T2newbb[ig] = T2newbb[ig - 1] + (T2newbb[ig - 2] - T2newbb[ig - 1])*fac
        converged = False
        nl1 = numpy.sqrt(float(T1newa[ig].size))
        nl2 = numpy.sqrt(float(T2newaa[ig].size))
        if iprint > 0:
            print("Time point {}".format(ig))
        i = 0
        while i < max_iter and not converged:
            # form new T1 and T2
            (T1a,T1b),(T2aa,T2ab,T2bb) = form_new_ampl_extrap_u(ig,method,Fa,Fb,Ia,Ib,Iabab,
                    T1newa[ig],T1newb[ig],T2newaa[ig],T2newab[ig],T2newbb[ig],
                    t1bara,t1barb,t2baraa,t2barab,t2barbb,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)

            res1 = numpy.linalg.norm(T1a - T1newa[ig]) / nl1
            res1 += numpy.linalg.norm(T1b - T1newb[ig]) / nl1
            res2 = numpy.linalg.norm(T2aa - T2newaa[ig]) / nl2
            res2 += numpy.linalg.norm(T2ab - T2newab[ig]) / nl2
            res2 += numpy.linalg.norm(T2bb - T2newbb[ig]) / nl2
            # damp new T-amplitudes
            T1newa[ig] = alpha*T1newa[ig] + (1.0 - alpha)*T1a.copy()
            T1newb[ig] = alpha*T1newb[ig] + (1.0 - alpha)*T1b.copy()
            T2newaa[ig] = alpha*T2newaa[ig] + (1.0 - alpha)*T2aa.copy()
            T2newab[ig] = alpha*T2newab[ig] + (1.0 - alpha)*T2ab.copy()
            T2newbb[ig] = alpha*T2newbb[ig] + (1.0 - alpha)*T2bb.copy()

            # determine convergence
            if iprint > 0:
                print(' %2d  %.4E' % (i+1,res1+res2))
            i = i + 1
            if res1 + res2 < thresh:
                converged = True
    return (T1newa,T1newb),(T2newaa,T2newab,T2newbb)

def ft_lambda_iter(method, L1old, L2old, T1, T2, F, I, D1, D2,
        g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    i = 0
    nl1 = numpy.linalg.norm(L1old) + 0.1
    nl2 = numpy.linalg.norm(L2old) + 0.1
    while i < max_iter and not converged:
        if method == "LCCSD":
            L1,L2 = ft_cc_equations.lccsd_lambda_simple(
                F,I,T1,T2,L1old,L2old,D1,D2,ti,ng,g,G,beta)
        elif method == "LCCD":
            L1 = L1old
            L2 = ft_cc_equations.lccd_lambda_simple(F,I,self.T2,
                    L2old,D2,ti,ng,g,G,beta)
        elif method == "CCSD":
            L1,L2 = ft_cc_equations.ccsd_lambda_opt(
                F,I,T1,T2,L1old,L2old,D1,D2,ti,ng,g,G,beta)
        elif method == "CCD":
            L1 = L1old
            L2 = ft_cc_equations.ccd_lambda_simple(F,I,T2,
                    L2old,D2,ti,ng,g,G,beta)
        else:
            raise Exception("Unrecognized method keyword")

        res1 = numpy.linalg.norm(L1 - L1old) / nl1
        res2 = numpy.linalg.norm(L2 - L2old) / nl2
        # compute new L-amplitudes
        L1old = alpha*L1old + (1.0 - alpha)*L1
        L2old = alpha*L2old + (1.0 - alpha)*L2
        nl1 = numpy.linalg.norm(L1old) + 0.1
        nl2 = numpy.linalg.norm(L2old) + 0.1
        L1 = None
        L2 = None

        # determine convergence
        if iprint > 0:
            print(' %2d  %.10f' % (i+1,res1 + res2))
        i = i + 1
        if res1 + res2 < thresh:
            converged = True

    if not converged:
        print("WARNING: CCSD Lambda-equations did not converge!")

    tend = time.time()
    if iprint > 0:
        print("Total CCSD Lambda time: %f s" % (tend - tbeg))

    return L1old,L2old

def ft_ulambda_iter(method, L1ain, L1bin, L2aain, L2abin, L2bbin, T1aold, T1bold,
        T2aaold, T2abold, T2bbold, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
        g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    na = D1a.shape[0]
    nb = D1b.shape[0]
    n = na + nb
    i = 0
    L1aold = L1ain
    L1bold = L1bin
    L2aaold = L2aain
    L2abold = L2abin
    L2bbold = L2bbin

    nl1 = numpy.linalg.norm(L1aold) + numpy.linalg.norm(L1bold) + 0.1
    nl2 = numpy.linalg.norm(L2aaold) + 0.1
    nl2 += numpy.linalg.norm(L2bbold)
    nl2 += 4*numpy.linalg.norm(L2abold)
    while i < max_iter and not converged:
        if method == "LCCSD":
            raise Exception("U-LCCSD lambda equations not implemented")
        elif method == "LCCD":
            raise Exception("U-LCCD lambda equations not implemented")
        elif method == "CCSD":
            L1a,L1b,L2aa,L2ab,L2bb = ft_cc_equations.uccsd_lambda_opt(
                Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,T2aaold,T2abold,T2bbold,
                L1aold,L1bold,L2aaold,L2abold,L2bbold,D1a,D1b,D2aa,D2ab,D2bb,
                ti,ng,g,G,beta)
        elif method == "CCD":
            raise Exception("UCCD lambda equations not implemented")
        else:
            raise Exception("Unrecognized method keyword")

        res1 = numpy.linalg.norm(L1a - L1aold) / nl1
        res1 += numpy.linalg.norm(L1b - L1bold) / nl1
        res2 = numpy.linalg.norm(L2aa - L2aaold) / nl2
        res2 += numpy.linalg.norm(L2ab - L2abold) / nl2
        res2 += numpy.linalg.norm(L2bb - L2bbold) / nl2
        # compute new L-amplitudes
        L1aold = alpha*L1aold + (1.0 - alpha)*L1a
        L1bold = alpha*L1bold + (1.0 - alpha)*L1b
        L2aaold = alpha*L2aaold + (1.0 - alpha)*L2aa
        L2abold = alpha*L2abold + (1.0 - alpha)*L2ab
        L2bbold = alpha*L2bbold + (1.0 - alpha)*L2bb
        nl1 = numpy.linalg.norm(L1aold) + numpy.linalg.norm(L1bold) + 0.1
        nl2 = numpy.linalg.norm(L2aaold) + 0.1
        nl2 += numpy.linalg.norm(L2bbold)
        nl2 += 4*numpy.linalg.norm(L2abold)
        L1a = None
        L1b = None
        L2aa = None
        L2ab = None
        L2bb = None

        # determine convergence
        if iprint > 0:
            print(' %2d  %.10f' % (i+1,res1 + res2))
        i = i + 1
        if res1 + res2 < thresh:
            converged = True

    if not converged:
        print("WARNING: CCSD Lambda-equations did not converge!")

    tend = time.time()
    if iprint > 0:
        print("Total CCSD Lambda time: %f s" % (tend - tbeg))

    return L1aold,L1bold,L2aaold,L2abold,L2bbold

def ft_integrals(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)

        # get FT fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        # get ERIs
        eri = sys.g_aint_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,i,j->ij',fmo,sfo,sfo)
        Fov = einsum('ia,i,a->ia',fmo,sfo,sfv)
        Fvo = einsum('ai,a,i->ai',fmo,sfv,sfo)
        Fvv = einsum('ab,a,b->ab',fmo,sfv,sfv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,sfv,sfv)
        Ivvvo = einsum('abci,a,b,c,i->abci',eri,sfv,sfv,sfv,sfo)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,sfv,sfv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri,sfv,sfv,sfo,sfo)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,sfv,sfo)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,sfv,sfv)
        Ivooo = einsum('akij,a,k,i,j->akij',eri,sfv,sfo,sfo,sfo)
        Iooov = einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,sfo,sfv)
        Ioooo = einsum('klij,k,l,i,j->klij',eri,sfo,sfo,sfo,sfo)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return F,I

def get_ft_integrals_neq(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis
        including real-time component."""
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get FT fock matrix
        fmo = sys.g_fock_tot(direc='f')
        fmo = (fmo - numpy.diag(en)).astype(complex)

        # pre-contract with fermi factors
        Foo = einsum('ij,j->ij',fmo[0],fo)
        Fvo = einsum('ai,a,i->ai',fmo[0],fv,fo)
        Fvv = einsum('ab,a->ab',fmo[0],fv)
        F = one_e_blocks(Foo,fmo[0],Fvo,Fvv)

        Foo = einsum('yij,j->yij',fmo,fo)
        Fvo = einsum('yai,a,i->yai',fmo,fv,fo)
        Fvv = einsum('yab,a->yab',fmo,fv)
        Ff = one_e_blocks(Foo,fmo,Fvo,Fvv)

        fmo = sys.g_fock_tot(direc='b')
        fmo = (fmo - numpy.diag(en)).astype(complex)
        Foo = einsum('yij,j->yij',fmo,fo)
        Fvo = einsum('yai,a,i->yai',fmo,fv,fo)
        Fvv = einsum('yab,a->yab',fmo,fv)
        Fb = one_e_blocks(Foo,fmo,Fvo,Fvv)

        # get ERIs
        eri = sys.g_aint_tot().astype(complex)

        Ivvvv = einsum('abcd,a,b->abcd',eri,fv,fv)
        Ivvvo = einsum('abci,a,b,i->abci',eri,fv,fv,fo)
        Ivovv = einsum('aibc,a->aibc',eri,fv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri,fv,fv,fo,fo)
        Ivovo = einsum('ajbi,a,i->ajbi',eri,fv,fo)
        Ivooo = einsum('akij,a,i,j->akij',eri,fv,fo,fo)
        Iooov = einsum('jkia,i->jkia',eri,fo)
        Ioooo = einsum('klij,i,j->klij',eri,fo,fo)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=eri,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return F,Ff,Fb,I

def uft_integrals(sys, ea, eb, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        na = ea.shape[0]
        nb = eb.shape[0]
        #en = numpy.concatenate((ea,eb))
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        sfoa = numpy.sqrt(foa)
        sfva = numpy.sqrt(fva)
        sfob = numpy.sqrt(fob)
        sfvb = numpy.sqrt(fvb)

        # get FT fock matrix
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)

        # pre-contract with fermi factors
        Fooa = einsum('ij,i,j->ij',fa,sfoa,sfoa)
        Fova = einsum('ia,i,a->ia',fa,sfoa,sfva)
        Fvoa = einsum('ai,a,i->ai',fa,sfva,sfoa)
        Fvva = einsum('ab,a,b->ab',fa,sfva,sfva)
        Fa = one_e_blocks(Fooa,Fova,Fvoa,Fvva)

        Foob = einsum('ij,i,j->ij',fb,sfob,sfob)
        Fovb = einsum('ia,i,a->ia',fb,sfob,sfvb)
        Fvob = einsum('ai,a,i->ai',fb,sfvb,sfob)
        Fvvb = einsum('ab,a,b->ab',fb,sfvb,sfvb)
        Fb = one_e_blocks(Foob,Fovb,Fvob,Fvvb)

        # get ERIs
        eriA,eriB,eriAB = sys.u_aint_tot()
        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriA,sfva,sfva,sfva,sfva)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriA,sfva,sfva,sfva,sfoa)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriA,sfva,sfoa,sfva,sfva)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriA,sfva,sfva,sfoa,sfoa)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriA,sfoa,sfoa,sfva,sfva)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriA,sfva,sfoa,sfva,sfoa)
        Ivooo = einsum('akij,a,k,i,j->akij',eriA,sfva,sfoa,sfoa,sfoa)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriA,sfoa,sfoa,sfoa,sfva)
        Ioooo = einsum('klij,k,l,i,j->klij',eriA,sfoa,sfoa,sfoa,sfoa)
        Ia = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriB,sfvb,sfvb,sfvb,sfvb)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriB,sfvb,sfvb,sfvb,sfob)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriB,sfvb,sfob,sfvb,sfvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriB,sfvb,sfvb,sfob,sfob)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriB,sfob,sfob,sfvb,sfvb)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriB,sfvb,sfob,sfvb,sfob)
        Ivooo = einsum('akij,a,k,i,j->akij',eriB,sfvb,sfob,sfob,sfob)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriB,sfob,sfob,sfob,sfvb)
        Ioooo = einsum('klij,k,l,i,j->klij',eriB,sfob,sfob,sfob,sfob)
        Ib = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriAB,sfva,sfvb,sfva,sfvb)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriAB,sfva,sfvb,sfva,sfob)
        Ivvov = einsum('abic,a,b,i,c->abic',eriAB,sfva,sfvb,sfoa,sfvb)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriAB,sfva,sfob,sfva,sfvb)
        Iovvv = einsum('iabc,i,a,b,c->iabc',eriAB,sfoa,sfvb,sfva,sfvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriAB,sfva,sfvb,sfoa,sfob)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriAB,sfva,sfob,sfva,sfob)
        Iovvo = einsum('jabi,j,a,b,i->jabi',eriAB,sfoa,sfvb,sfva,sfob)
        Ivoov = einsum('ajib,a,j,i,b->ajib',eriAB,sfva,sfob,sfoa,sfvb)
        Iovov = einsum('jaib,j,a,i,b->jaib',eriAB,sfoa,sfvb,sfoa,sfvb)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriAB,sfoa,sfob,sfva,sfvb)
        Ivooo = einsum('akij,a,k,i,j->akij',eriAB,sfva,sfob,sfoa,sfob)
        Iovoo = einsum('kaij,k,a,i,j->kaij',eriAB,sfoa,sfvb,sfoa,sfob)
        Ioovo = einsum('jkai,j,k,a,i->jkai',eriAB,sfoa,sfob,sfva,sfob)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriAB,sfoa,sfob,sfoa,sfvb)
        Ioooo = einsum('klij,k,l,i,j->klij',eriAB,sfoa,sfob,sfoa,sfob)
        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)

        return Fa,Fb,Ia,Ib,Iabab

def rft_integrals(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)

        # get FT fock matrix
        fmo = sys.r_fock_tot()
        fmo = fmo - numpy.diag(en)

        # get ERIs
        eri = sys.r_int_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,i,j->ij',fmo,sfo,sfo)
        Fov = einsum('ia,i,a->ia',fmo,sfo,sfv)
        Fvo = einsum('ai,a,i->ai',fmo,sfv,sfo)
        Fvv = einsum('ab,a,b->ab',fmo,sfv,sfv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,sfv,sfv)
        Ivvvo = einsum('abci,a,b,c,i->abci',eri,sfv,sfv,sfv,sfo)
        Ivvov = einsum('abic,a,b,i,c->abic',eri,sfv,sfv,sfo,sfv)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,sfv,sfv)
        Iovvv = einsum('iabc,i,a,b,c->iabc',eri,sfo,sfv,sfv,sfv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri,sfv,sfv,sfo,sfo)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,sfv,sfo)
        Iovvo = einsum('jabi,j,a,b,i->jabi',eri,sfo,sfv,sfv,sfo)
        Ivoov = einsum('ajib,a,j,i,b->ajib',eri,sfv,sfo,sfo,sfv)
        Iovov = einsum('jaib,j,a,i,b->jaib',eri,sfo,sfv,sfo,sfv)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,sfv,sfv)
        Ivooo = einsum('akij,a,k,i,j->akij',eri,sfv,sfo,sfo,sfo)
        Iovoo = einsum('kaij,k,a,i,j->kaij',eri,sfo,sfv,sfo,sfo)
        Ioovo = einsum('jkai,j,k,a,i->jkai',eri,sfo,sfo,sfv,sfo)
        Iooov = einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,sfo,sfv)
        Ioooo = einsum('klij,k,l,i,j->klij',eri,sfo,sfo,sfo,sfo)
        I = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)
        return F,I

def ft_active_integrals(sys, en, focc, fvir, iocc, ivir):
        """Return one and two-electron integrals in the general spin orbital basis
        with small occupations excluded."""
        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        sfo = numpy.sqrt(focc)
        sfv = numpy.sqrt(fvir)

        # get ERIs
        eri = sys.g_aint_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,i,j->ij',fmo[numpy.ix_(iocc,iocc)],sfo,sfo)
        Fov = einsum('ia,i,a->ia',fmo[numpy.ix_(iocc,ivir)],sfo,sfv)
        Fvo = einsum('ai,a,i->ai',fmo[numpy.ix_(ivir,iocc)],sfv,sfo)
        Fvv = einsum('ab,a,b->ab',fmo[numpy.ix_(ivir,ivir)],sfv,sfv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],sfv,sfv,sfv,sfv)
        Ivvvo = einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],sfv,sfv,sfv,sfo)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],sfv,sfo,sfv,sfv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],sfv,sfv,sfo,sfo)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],sfo,sfo,sfv,sfv)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],sfv,sfo,sfv,sfo)
        Ivooo = einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],sfv,sfo,sfo,sfo)
        Iooov = einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],sfo,sfo,sfo,sfv)
        Ioooo = einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],sfo,sfo,sfo,sfo)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        return F,I

def uft_active_integrals(sys, ea, eb, foa, fva, fob, fvb, iocca, ivira, ioccb, ivirb):
        """Return one and two-electron integrals in the general spin orbital basis
        with small occupations excluded."""
        # get FT Fock matrix
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)

        sfoa = numpy.sqrt(foa)
        sfva = numpy.sqrt(fva)
        sfob = numpy.sqrt(fob)
        sfvb = numpy.sqrt(fvb)

        # pre-contract with fermi factors
        Fooa = einsum('ij,i,j->ij',fa[numpy.ix_(iocca,iocca)],sfoa,sfoa)
        Fova = einsum('ia,i,a->ia',fa[numpy.ix_(iocca,ivira)],sfoa,sfva)
        Fvoa = einsum('ai,a,i->ai',fa[numpy.ix_(ivira,iocca)],sfva,sfoa)
        Fvva = einsum('ab,a,b->ab',fa[numpy.ix_(ivira,ivira)],sfva,sfva)
        Fa = one_e_blocks(Fooa,Fova,Fvoa,Fvva)

        Foob = einsum('ij,i,j->ij',fb[numpy.ix_(ioccb,ioccb)],sfob,sfob)
        Fovb = einsum('ia,i,a->ia',fb[numpy.ix_(ioccb,ivirb)],sfob,sfvb)
        Fvob = einsum('ai,a,i->ai',fb[numpy.ix_(ivirb,ioccb)],sfvb,sfob)
        Fvvb = einsum('ab,a,b->ab',fb[numpy.ix_(ivirb,ivirb)],sfvb,sfvb)
        Fb = one_e_blocks(Foob,Fovb,Fvob,Fvvb)

        # get ERIs
        eriA,eriB,eriAB = sys.u_aint_tot()

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriA[numpy.ix_(ivira,ivira,ivira,ivira)],sfva,sfva,sfva,sfva)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriA[numpy.ix_(ivira,ivira,ivira,iocca)],sfva,sfva,sfva,sfoa)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriA[numpy.ix_(ivira,iocca,ivira,ivira)],sfva,sfoa,sfva,sfva)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriA[numpy.ix_(ivira,ivira,iocca,iocca)],sfva,sfva,sfoa,sfoa)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriA[numpy.ix_(iocca,iocca,ivira,ivira)],sfoa,sfoa,sfva,sfva)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriA[numpy.ix_(ivira,iocca,ivira,iocca)],sfva,sfoa,sfva,sfoa)
        Ivooo = einsum('akij,a,k,i,j->akij',eriA[numpy.ix_(ivira,iocca,iocca,iocca)],sfva,sfoa,sfoa,sfoa)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriA[numpy.ix_(iocca,iocca,iocca,ivira)],sfoa,sfoa,sfoa,sfva)
        Ioooo = einsum('klij,k,l,i,j->klij',eriA[numpy.ix_(iocca,iocca,iocca,iocca)],sfoa,sfoa,sfoa,sfoa)
        Ia = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriB[numpy.ix_(ivirb,ivirb,ivirb,ivirb)],sfvb,sfvb,sfvb,sfvb)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriB[numpy.ix_(ivirb,ivirb,ivirb,ioccb)],sfvb,sfvb,sfvb,sfob)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriB[numpy.ix_(ivirb,ioccb,ivirb,ivirb)],sfvb,sfob,sfvb,sfvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriB[numpy.ix_(ivirb,ivirb,ioccb,ioccb)],sfvb,sfvb,sfob,sfob)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriB[numpy.ix_(ioccb,ioccb,ivirb,ivirb)],sfob,sfob,sfvb,sfvb)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriB[numpy.ix_(ivirb,ioccb,ivirb,ioccb)],sfvb,sfob,sfvb,sfob)
        Ivooo = einsum('akij,a,k,i,j->akij',eriB[numpy.ix_(ivirb,ioccb,ioccb,ioccb)],sfvb,sfob,sfob,sfob)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriB[numpy.ix_(ioccb,ioccb,ioccb,ivirb)],sfob,sfob,sfob,sfvb)
        Ioooo = einsum('klij,k,l,i,j->klij',eriB[numpy.ix_(ioccb,ioccb,ioccb,ioccb)],sfob,sfob,sfob,sfob)
        Ib = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriAB[numpy.ix_(ivira,ivirb,ivira,ivirb)],sfva,sfvb,sfva,sfvb)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriAB[numpy.ix_(ivira,ivirb,ivira,ioccb)],sfva,sfvb,sfva,sfob)
        Ivvov = einsum('abic,a,b,i,c->abic',eriAB[numpy.ix_(ivira,ivirb,iocca,ivirb)],sfva,sfvb,sfoa,sfvb)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriAB[numpy.ix_(ivira,ioccb,ivira,ivirb)],sfva,sfob,sfva,sfvb)
        Iovvv = einsum('iabc,i,a,b,c->iabc',eriAB[numpy.ix_(iocca,ivirb,ivira,ivirb)],sfoa,sfvb,sfva,sfvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriAB[numpy.ix_(ivira,ivirb,iocca,ioccb)],sfva,sfvb,sfoa,sfob)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriAB[numpy.ix_(ivira,ioccb,ivira,ioccb)],sfva,sfob,sfva,sfob)
        Iovvo = einsum('jabi,j,a,b,i->jabi',eriAB[numpy.ix_(iocca,ivirb,ivira,ioccb)],sfoa,sfvb,sfva,sfob)
        Ivoov = einsum('ajib,a,j,i,b->ajib',eriAB[numpy.ix_(ivira,ioccb,iocca,ivirb)],sfva,sfob,sfoa,sfvb)
        Iovov = einsum('jaib,j,a,i,b->jaib',eriAB[numpy.ix_(iocca,ivirb,iocca,ivirb)],sfoa,sfvb,sfoa,sfvb)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriAB[numpy.ix_(iocca,ioccb,ivira,ivirb)],sfoa,sfob,sfva,sfvb)
        Ivooo = einsum('akij,a,k,i,j->akij',eriAB[numpy.ix_(ivira,ioccb,iocca,ioccb)],sfva,sfob,sfoa,sfob)
        Iovoo = einsum('kaij,k,a,i,j->kaij',eriAB[numpy.ix_(iocca,ivirb,iocca,ioccb)],sfoa,sfvb,sfoa,sfob)
        Ioovo = einsum('jkai,j,k,a,i->jkai',eriAB[numpy.ix_(iocca,ioccb,ivira,ioccb)],sfoa,sfob,sfva,sfob)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriAB[numpy.ix_(iocca,ioccb,iocca,ivirb)],sfoa,sfob,sfoa,sfvb)
        Ioooo = einsum('klij,k,l,i,j->klij',eriAB[numpy.ix_(iocca,ioccb,iocca,ioccb)],sfoa,sfob,sfoa,sfob)
        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)

        return Fa,Fb,Ia,Ib,Iabab

def rft_active_integrals(sys, en, focc, fvir, iocc, ivir):
        """Return one and two-electron integrals in the general spin orbital basis."""
        # get FT fock matrix
        fmo = sys.r_fock_tot()
        fmo = fmo - numpy.diag(en)

        # get ERIs
        eri = sys.r_int_tot()

        # square root of occupation numbers
        sfo = numpy.sqrt(focc)
        sfv = numpy.sqrt(fvir)

        # pre-contract with fermi factors
        Foo = einsum('ij,i,j->ij',fmo[numpy.ix_(iocc,iocc)],sfo,sfo)
        Fov = einsum('ia,i,a->ia',fmo[numpy.ix_(iocc,ivir)],sfo,sfv)
        Fvo = einsum('ai,a,i->ai',fmo[numpy.ix_(ivir,iocc)],sfv,sfo)
        Fvv = einsum('ab,a,b->ab',fmo[numpy.ix_(ivir,ivir)],sfv,sfv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],sfv,sfv,sfv,sfv)
        Ivvvo = einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],sfv,sfv,sfv,sfo)
        Ivvov = einsum('abic,a,b,i,c->abic',eri[numpy.ix_(ivir,ivir,iocc,ivir)],sfv,sfv,sfo,sfv)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],sfv,sfo,sfv,sfv)
        Iovvv = einsum('iabc,i,a,b,c->iabc',eri[numpy.ix_(iocc,ivir,ivir,ivir)],sfo,sfv,sfv,sfv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],sfv,sfv,sfo,sfo)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],sfv,sfo,sfv,sfo)
        Iovvo = einsum('jabi,j,a,b,i->jabi',eri[numpy.ix_(iocc,ivir,ivir,iocc)],sfo,sfv,sfv,sfo)
        Ivoov = einsum('ajib,a,j,i,b->ajib',eri[numpy.ix_(ivir,iocc,iocc,ivir)],sfv,sfo,sfo,sfv)
        Iovov = einsum('jaib,j,a,i,b->jaib',eri[numpy.ix_(iocc,ivir,iocc,ivir)],sfo,sfv,sfo,sfv)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],sfo,sfo,sfv,sfv)
        Ivooo = einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],sfv,sfo,sfo,sfo)
        Iovoo = einsum('kaij,k,a,i,j->kaij',eri[numpy.ix_(iocc,ivir,iocc,iocc)],sfo,sfv,sfo,sfo)
        Ioovo = einsum('jkai,j,k,a,i->jkai',eri[numpy.ix_(iocc,iocc,ivir,iocc)],sfo,sfo,sfv,sfo)
        Iooov = einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],sfo,sfo,sfo,sfv)
        Ioooo = einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],sfo,sfo,sfo,sfo)
        I = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)
        return F,I

def _form_ft_d_eris(eri, sfo, sfv, dso, dsv):
        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri,dsv,sfv,sfv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri,sfv,dsv,sfv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,dsv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,sfv,dsv)

        Ivvvo = einsum('abci,a,b,c,i->abci',eri,dsv,sfv,sfv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri,sfv,dsv,sfv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri,sfv,sfv,dsv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri,sfv,sfv,sfv,dso)

        Ivovv = einsum('aibc,a,i,b,c->aibc',eri,dsv,sfo,sfv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri,sfv,dso,sfv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,dsv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,sfv,dsv)

        Ivvoo = einsum('abij,a,b,i,j->abij',eri,dsv,sfv,sfo,sfo)\
              + einsum('abij,a,b,i,j->abij',eri,sfv,dsv,sfo,sfo)\
              + einsum('abij,a,b,i,j->abij',eri,sfv,sfv,dso,sfo)\
              + einsum('abij,a,b,i,j->abij',eri,sfv,sfv,sfo,dso)

        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri,dsv,sfo,sfv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri,sfv,dso,sfv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,dsv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,sfv,dso)

        Ioovv = einsum('ijab,i,j,a,b->ijab',eri,dso,sfo,sfv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri,sfo,dso,sfv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,dsv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,sfv,dsv)

        Ivooo = einsum('akij,a,k,i,j->akij',eri,dsv,sfo,sfo,sfo)\
              + einsum('akij,a,k,i,j->akij',eri,sfv,dso,sfo,sfo)\
              + einsum('akij,a,k,i,j->akij',eri,sfv,sfo,dso,sfo)\
              + einsum('akij,a,k,i,j->akij',eri,sfv,sfo,sfo,dso)

        Iooov = einsum('jkia,j,k,i,a->jkia',eri,dso,sfo,sfo,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri,sfo,dso,sfo,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,dso,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,sfo,dsv)

        Ioooo = einsum('klij,k,l,i,j->klij',eri,dso,sfo,sfo,sfo)\
              + einsum('klij,k,l,i,j->klij',eri,sfo,dso,sfo,sfo)\
              + einsum('klij,k,l,i,j->klij',eri,sfo,sfo,dso,sfo)\
              + einsum('klij,k,l,i,j->klij',eri,sfo,sfo,sfo,dso)

        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return I

def _form_ft_d_active_eris(eri, sfo, sfv, dso, dsv, iocc, ivir):
        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],dsv,sfv,sfv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],sfv,dsv,sfv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],sfv,sfv,dsv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],sfv,sfv,sfv,dsv)

        Ivvvo = einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],dsv,sfv,sfv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],sfv,dsv,sfv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],sfv,sfv,dsv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],sfv,sfv,sfv,dso)

        Ivovv = einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],dsv,sfo,sfv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],sfv,dso,sfv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],sfv,sfo,dsv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],sfv,sfo,sfv,dsv)

        Ivvoo = einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],dsv,sfv,sfo,sfo)\
              + einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],sfv,dsv,sfo,sfo)\
              + einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],sfv,sfv,dso,sfo)\
              + einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],sfv,sfv,sfo,dso)

        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],dsv,sfo,sfv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],sfv,dso,sfv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],sfv,sfo,dsv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],sfv,sfo,sfv,dso)

        Ioovv = einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],dso,sfo,sfv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],sfo,dso,sfv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],sfo,sfo,dsv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],sfo,sfo,sfv,dsv)

        Ivooo = einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],dsv,sfo,sfo,sfo)\
              + einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],sfv,dso,sfo,sfo)\
              + einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],sfv,sfo,dso,sfo)\
              + einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],sfv,sfo,sfo,dso)

        Iooov = einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],dso,sfo,sfo,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],sfo,dso,sfo,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],sfo,sfo,dso,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],sfo,sfo,sfo,dsv)

        Ioooo = einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],dso,sfo,sfo,sfo)\
              + einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],sfo,dso,sfo,sfo)\
              + einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],sfo,sfo,dso,sfo)\
              + einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],sfo,sfo,sfo,dso)

        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return I

def ft_d_integrals(sys, en, fo, fv, dvec):
        """form integrals contracted with derivatives of occupation numbers in the
        spin-orbital basis."""

        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)
        fd = sys.g_fock_d_tot(dvec)

        # get ERIs
        eri = sys.g_aint_tot()
        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)
        dso = -0.5*sfo*fv*dvec
        dsv = +0.5*sfv*fo*dvec

        # form derivative integrals
        Foo = einsum('ij,i,j->ij',fd,sfo,sfo)\
                + einsum('ij,i,j->ij',fmo,dso,sfo)\
                + einsum('ij,i,j->ij',fmo,sfo,dso)
        Fov = einsum('ia,i,a->ia',fd,sfo,sfv)\
                + einsum('ia,i,a->ia',fmo,dso,sfv)\
                + einsum('ia,i,a->ia',fmo,sfo,dsv)
        Fvo = einsum('ai,a,i->ai',fd,sfv,sfo)\
                + einsum('ai,a,i->ai',fmo,dsv,sfo)\
                + einsum('ai,a,i->ai',fmo,sfv,dso)
        Fvv = einsum('ab,a,b->ab',fd,sfv,sfv)\
                + einsum('ab,a,b->ab',fmo,dsv,sfv)\
                + einsum('ab,a,b->ab',fmo,sfv,dsv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)
        
        I = _form_ft_d_eris(eri,sfo,sfv,dso,dsv)
        return F,I

def u_ft_d_integrals(sys, ea, eb, foa, fva, fob, fvb, dveca, dvecb):
        """form unrestricted integrals contracted with derivatives of occupation numbers."""
        na = ea.shape[0]
        nb = eb.shape[0]

        # get FT Fock matrices
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)
        fda,fdb = sys.u_fock_d_tot(dveca,dvecb)

        sfoa = numpy.sqrt(foa)
        sfva = numpy.sqrt(fva)
        dsoa = -0.5*sfoa*fva*dveca
        dsva = +0.5*sfva*foa*dveca

        sfob = numpy.sqrt(fob)
        sfvb = numpy.sqrt(fvb)
        dsob = -0.5*sfob*fvb*dvecb
        dsvb = +0.5*sfvb*fob*dvecb

        Fooa = einsum('ij,i,j->ij',fda,sfoa,sfoa)\
                + einsum('ij,i,j->ij',fa,dsoa,sfoa)\
                + einsum('ij,i,j->ij',fa,sfoa,dsoa)
        Fova = einsum('ia,i,a->ia',fda,sfoa,sfva)\
                + einsum('ia,i,a->ia',fa,dsoa,sfva)\
                + einsum('ia,i,a->ia',fa,sfoa,dsva)
        Fvoa = einsum('ai,a,i->ai',fda,sfva,sfoa)\
                + einsum('ai,a,i->ai',fa,dsva,sfoa)\
                + einsum('ai,a,i->ai',fa,sfva,dsoa)
        Fvva = einsum('ab,a,b->ab',fda,sfva,sfva)\
                + einsum('ab,a,b->ab',fa,dsva,sfva)\
                + einsum('ab,a,b->ab',fa,sfva,dsva)
        Fa = one_e_blocks(Fooa,Fova,Fvoa,Fvva)

        Foob = einsum('ij,i,j->ij',fdb,sfob,sfob)\
                + einsum('ij,i,j->ij',fb,dsob,sfob)\
                + einsum('ij,i,j->ij',fb,sfob,dsob)
        Fovb = einsum('ia,i,a->ia',fdb,sfob,sfvb)\
                + einsum('ia,i,a->ia',fb,dsob,sfvb)\
                + einsum('ia,i,a->ia',fb,sfob,dsvb)
        Fvob = einsum('ai,a,i->ai',fdb,sfvb,sfob)\
                + einsum('ai,a,i->ai',fb,dsvb,sfob)\
                + einsum('ai,a,i->ai',fb,sfvb,dsob)
        Fvvb = einsum('ab,a,b->ab',fdb,sfvb,sfvb)\
                + einsum('ab,a,b->ab',fb,dsvb,sfvb)\
                + einsum('ab,a,b->ab',fb,sfvb,dsvb)
        Fb = one_e_blocks(Foob,Fovb,Fvob,Fvvb)

        # get ERIs
        Ia,Ib,Iabab = sys.u_aint_tot()

        Ia = _form_ft_d_eris(Ia,sfoa,sfva,dsoa,dsva)
        Ib = _form_ft_d_eris(Ib,sfob,sfvb,dsob,dsvb)

        Ivvvv =  einsum('abcd,a,b,c,d->abcd',Iabab,dsva,sfvb,sfva,sfvb)
        Ivvvv += einsum('abcd,a,b,c,d->abcd',Iabab,sfva,dsvb,sfva,sfvb)
        Ivvvv += einsum('abcd,a,b,c,d->abcd',Iabab,sfva,sfvb,dsva,sfvb)
        Ivvvv += einsum('abcd,a,b,c,d->abcd',Iabab,sfva,sfvb,sfva,dsvb)

        Ivvvo =  einsum('abci,a,b,c,i->abci',Iabab,dsva,sfvb,sfva,sfob)
        Ivvvo += einsum('abci,a,b,c,i->abci',Iabab,sfva,dsvb,sfva,sfob)
        Ivvvo += einsum('abci,a,b,c,i->abci',Iabab,sfva,sfvb,dsva,sfob)
        Ivvvo += einsum('abci,a,b,c,i->abci',Iabab,sfva,sfvb,sfva,dsob)

        Ivvov =  einsum('abic,a,b,i,c->abic',Iabab,dsva,sfvb,sfoa,sfvb)
        Ivvov += einsum('abic,a,b,i,c->abic',Iabab,sfva,dsvb,sfoa,sfvb)
        Ivvov += einsum('abic,a,b,i,c->abic',Iabab,sfva,sfvb,dsoa,sfvb)
        Ivvov += einsum('abic,a,b,i,c->abic',Iabab,sfva,sfvb,sfoa,dsvb)

        Ivovv =  einsum('aibc,a,i,b,c->aibc',Iabab,dsva,sfob,sfva,sfvb)
        Ivovv += einsum('aibc,a,i,b,c->aibc',Iabab,sfva,dsob,sfva,sfvb)
        Ivovv += einsum('aibc,a,i,b,c->aibc',Iabab,sfva,sfob,dsva,sfvb)
        Ivovv += einsum('aibc,a,i,b,c->aibc',Iabab,sfva,sfob,sfva,dsvb)

        Iovvv =  einsum('iabc,i,a,b,c->iabc',Iabab,dsoa,sfvb,sfva,sfvb)
        Iovvv += einsum('iabc,i,a,b,c->iabc',Iabab,sfoa,dsvb,sfva,sfvb)
        Iovvv += einsum('iabc,i,a,b,c->iabc',Iabab,sfoa,sfvb,dsva,sfvb)
        Iovvv += einsum('iabc,i,a,b,c->iabc',Iabab,sfoa,sfvb,sfva,dsvb)

        Ivvoo =  einsum('abij,a,b,i,j->abij',Iabab,dsva,sfvb,sfoa,sfob)
        Ivvoo += einsum('abij,a,b,i,j->abij',Iabab,sfva,dsvb,sfoa,sfob)
        Ivvoo += einsum('abij,a,b,i,j->abij',Iabab,sfva,sfvb,dsoa,sfob)
        Ivvoo += einsum('abij,a,b,i,j->abij',Iabab,sfva,sfvb,sfoa,dsob)

        Ivovo =  einsum('ajbi,a,j,b,i->ajbi',Iabab,dsva,sfob,sfva,sfob)
        Ivovo += einsum('ajbi,a,j,b,i->ajbi',Iabab,sfva,dsob,sfva,sfob)
        Ivovo += einsum('ajbi,a,j,b,i->ajbi',Iabab,sfva,sfob,dsva,sfob)
        Ivovo += einsum('ajbi,a,j,b,i->ajbi',Iabab,sfva,sfob,sfva,dsob)

        Iovvo =  einsum('jabi,j,a,b,i->jabi',Iabab,dsoa,sfvb,sfva,sfob)
        Iovvo += einsum('jabi,j,a,b,i->jabi',Iabab,sfoa,dsvb,sfva,sfob)
        Iovvo += einsum('jabi,j,a,b,i->jabi',Iabab,sfoa,sfvb,dsva,sfob)
        Iovvo += einsum('jabi,j,a,b,i->jabi',Iabab,sfoa,sfvb,sfva,dsob)

        Ivoov =  einsum('ajib,a,j,i,b->ajib',Iabab,dsva,sfob,sfoa,sfvb)
        Ivoov += einsum('ajib,a,j,i,b->ajib',Iabab,sfva,dsob,sfoa,sfvb)
        Ivoov += einsum('ajib,a,j,i,b->ajib',Iabab,sfva,sfob,dsoa,sfvb)
        Ivoov += einsum('ajib,a,j,i,b->ajib',Iabab,sfva,sfob,sfoa,dsvb)

        Iovov =  einsum('jaib,j,a,i,b->jaib',Iabab,dsoa,sfvb,sfoa,sfvb)
        Iovov += einsum('jaib,j,a,i,b->jaib',Iabab,sfoa,dsvb,sfoa,sfvb)
        Iovov += einsum('jaib,j,a,i,b->jaib',Iabab,sfoa,sfvb,dsoa,sfvb)
        Iovov += einsum('jaib,j,a,i,b->jaib',Iabab,sfoa,sfvb,sfoa,dsvb)

        Ioovv =  einsum('ijab,i,j,a,b->ijab',Iabab,dsoa,sfob,sfva,sfvb)
        Ioovv += einsum('ijab,i,j,a,b->ijab',Iabab,sfoa,dsob,sfva,sfvb)
        Ioovv += einsum('ijab,i,j,a,b->ijab',Iabab,sfoa,sfob,dsva,sfvb)
        Ioovv += einsum('ijab,i,j,a,b->ijab',Iabab,sfoa,sfob,sfva,dsvb)

        Ivooo =  einsum('akij,a,k,i,j->akij',Iabab,dsva,sfob,sfoa,sfob)
        Ivooo += einsum('akij,a,k,i,j->akij',Iabab,sfva,dsob,sfoa,sfob)
        Ivooo += einsum('akij,a,k,i,j->akij',Iabab,sfva,sfob,dsoa,sfob)
        Ivooo += einsum('akij,a,k,i,j->akij',Iabab,sfva,sfob,sfoa,dsob)

        Iovoo =  einsum('kaij,k,a,i,j->kaij',Iabab,dsoa,sfvb,sfoa,sfob)
        Iovoo += einsum('kaij,k,a,i,j->kaij',Iabab,sfoa,dsvb,sfoa,sfob)
        Iovoo += einsum('kaij,k,a,i,j->kaij',Iabab,sfoa,sfvb,dsoa,sfob)
        Iovoo += einsum('kaij,k,a,i,j->kaij',Iabab,sfoa,sfvb,sfoa,dsob)

        Ioovo =  einsum('jkai,j,k,a,i->jkai',Iabab,dsoa,sfob,sfva,sfob)
        Ioovo += einsum('jkai,j,k,a,i->jkai',Iabab,sfoa,dsob,sfva,sfob)
        Ioovo += einsum('jkai,j,k,a,i->jkai',Iabab,sfoa,sfob,dsva,sfob)
        Ioovo += einsum('jkai,j,k,a,i->jkai',Iabab,sfoa,sfob,sfva,dsob)

        Iooov =  einsum('jkia,j,k,i,a->jkia',Iabab,dsoa,sfob,sfoa,sfvb)
        Iooov += einsum('jkia,j,k,i,a->jkia',Iabab,sfoa,dsob,sfoa,sfvb)
        Iooov += einsum('jkia,j,k,i,a->jkia',Iabab,sfoa,sfob,dsoa,sfvb)
        Iooov += einsum('jkia,j,k,i,a->jkia',Iabab,sfoa,sfob,sfoa,dsvb)

        Ioooo =  einsum('klij,k,l,i,j->klij',Iabab,dsoa,sfob,sfoa,sfob)
        Ioooo += einsum('klij,k,l,i,j->klij',Iabab,sfoa,dsob,sfoa,sfob)
        Ioooo += einsum('klij,k,l,i,j->klij',Iabab,sfoa,sfob,dsoa,sfob)
        Ioooo += einsum('klij,k,l,i,j->klij',Iabab,sfoa,sfob,sfoa,dsob)

        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)

        return Fa,Fb,Ia,Ib,Iabab

def ft_d_active_integrals(sys, en, fo, fv, iocc, ivir, dvec):
        """Return one and two-electron integrals in the general spin orbital basis
        with small occupations excluded."""
        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)
        fd = sys.g_fock_d_tot(dvec)

        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)
        dso = -0.5*sfo[numpy.ix_(iocc)]*fv[numpy.ix_(iocc)]*dvec[numpy.ix_(iocc)]
        dsv = +0.5*sfv[numpy.ix_(ivir)]*fo[numpy.ix_(ivir)]*dvec[numpy.ix_(ivir)]
        sfo = sfo[numpy.ix_(iocc)]
        sfv = sfv[numpy.ix_(ivir)]

        # form derivative integrals
        Foo = einsum('ij,i,j->ij',fd[numpy.ix_(iocc,iocc)],sfo,sfo)\
                + einsum('ij,i,j->ij',fmo[numpy.ix_(iocc,iocc)],dso,sfo)\
                + einsum('ij,i,j->ij',fmo[numpy.ix_(iocc,iocc)],sfo,dso)
        Fov = einsum('ia,i,a->ia',fd[numpy.ix_(iocc,ivir)],sfo,sfv)\
                + einsum('ia,i,a->ia',fmo[numpy.ix_(iocc,ivir)],dso,sfv)\
                + einsum('ia,i,a->ia',fmo[numpy.ix_(iocc,ivir)],sfo,dsv)
        Fvo = einsum('ai,a,i->ai',fd[numpy.ix_(ivir,iocc)],sfv,sfo)\
                + einsum('ai,a,i->ai',fmo[numpy.ix_(ivir,iocc)],dsv,sfo)\
                + einsum('ai,a,i->ai',fmo[numpy.ix_(ivir,iocc)],sfv,dso)
        Fvv = einsum('ab,a,b->ab',fd[numpy.ix_(ivir,ivir)],sfv,sfv)\
                + einsum('ab,a,b->ab',fmo[numpy.ix_(ivir,ivir)],dsv,sfv)\
                + einsum('ab,a,b->ab',fmo[numpy.ix_(ivir,ivir)],sfv,dsv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        # get ERIs
        eri = sys.g_aint_tot()
        I = _form_ft_d_active_eris(eri, sfo, sfv, dso, dsv, iocc, ivir)
        return F,I

def uft_d_active_integrals(
        sys, ea, eb, foa, fva, fob, fvb,
        iocca, ivira, ioccb, ivirb, dveca, dvecb):
        """Return derivatives of unrestricted one and two-electron integrals
        with small occupations excluded."""

        # get FT Fock matrix
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)
        fda,fdb = sys.u_fock_d_tot(dveca,dvecb)

        sfoa = numpy.sqrt(foa)
        sfva = numpy.sqrt(fva)
        sfob = numpy.sqrt(fob)
        sfvb = numpy.sqrt(fvb)
        dsoa = -0.5*(sfoa*fva*dveca)[numpy.ix_(iocca)]
        dsva = +0.5*(sfva*foa*dveca)[numpy.ix_(ivira)]
        dsob = -0.5*(sfob*fvb*dvecb)[numpy.ix_(ioccb)]
        dsvb = +0.5*(sfvb*fob*dvecb)[numpy.ix_(ivirb)]
        sfoa = sfoa[numpy.ix_(iocca)]
        sfva = sfva[numpy.ix_(ivira)]
        sfob = sfob[numpy.ix_(ioccb)]
        sfvb = sfvb[numpy.ix_(ivirb)]

        # form derivative integrals
        Foo = einsum('ij,i,j->ij',fda[numpy.ix_(iocca,iocca)],sfoa,sfoa)\
                + einsum('ij,i,j->ij',fa[numpy.ix_(iocca,iocca)],dsoa,sfoa)\
                + einsum('ij,i,j->ij',fa[numpy.ix_(iocca,iocca)],sfoa,dsoa)
        Fov = einsum('ia,i,a->ia',fda[numpy.ix_(iocca,ivira)],sfoa,sfva)\
                + einsum('ia,i,a->ia',fa[numpy.ix_(iocca,ivira)],dsoa,sfva)\
                + einsum('ia,i,a->ia',fa[numpy.ix_(iocca,ivira)],sfoa,dsva)
        Fvo = einsum('ai,a,i->ai',fda[numpy.ix_(ivira,iocca)],sfva,sfoa)\
                + einsum('ai,a,i->ai',fa[numpy.ix_(ivira,iocca)],dsva,sfoa)\
                + einsum('ai,a,i->ai',fa[numpy.ix_(ivira,iocca)],sfva,dsoa)
        Fvv = einsum('ab,a,b->ab',fda[numpy.ix_(ivira,ivira)],sfva,sfva)\
                + einsum('ab,a,b->ab',fa[numpy.ix_(ivira,ivira)],dsva,sfva)\
                + einsum('ab,a,b->ab',fa[numpy.ix_(ivira,ivira)],sfva,dsva)
        Fa = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Foo = einsum('ij,i,j->ij',fdb[numpy.ix_(ioccb,ioccb)],sfob,sfob)\
                + einsum('ij,i,j->ij',fb[numpy.ix_(ioccb,ioccb)],dsob,sfob)\
                + einsum('ij,i,j->ij',fb[numpy.ix_(ioccb,ioccb)],sfob,dsob)
        Fov = einsum('ia,i,a->ia',fdb[numpy.ix_(ioccb,ivirb)],sfob,sfvb)\
                + einsum('ia,i,a->ia',fb[numpy.ix_(ioccb,ivirb)],dsob,sfvb)\
                + einsum('ia,i,a->ia',fb[numpy.ix_(ioccb,ivirb)],sfob,dsvb)
        Fvo = einsum('ai,a,i->ai',fdb[numpy.ix_(ivirb,ioccb)],sfvb,sfob)\
                + einsum('ai,a,i->ai',fb[numpy.ix_(ivirb,ioccb)],dsvb,sfob)\
                + einsum('ai,a,i->ai',fb[numpy.ix_(ivirb,ioccb)],sfvb,dsob)
        Fvv = einsum('ab,a,b->ab',fdb[numpy.ix_(ivirb,ivirb)],sfvb,sfvb)\
                + einsum('ab,a,b->ab',fb[numpy.ix_(ivirb,ivirb)],dsvb,sfvb)\
                + einsum('ab,a,b->ab',fb[numpy.ix_(ivirb,ivirb)],sfvb,dsvb)
        Fb = one_e_blocks(Foo,Fov,Fvo,Fvv)

        # get ERIs
        eriA,eriB,eriAB = sys.u_aint_tot()
        Ia = _form_ft_d_active_eris(eriA, sfoa, sfva, dsoa, dsva, iocca, ivira)
        Ib = _form_ft_d_active_eris(eriB, sfob, sfvb, dsob, dsvb, ioccb, ivirb)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriAB[numpy.ix_(ivira,ivirb,ivira,ivirb)],dsva,sfvb,sfva,sfvb)\
              + einsum('abcd,a,b,c,d->abcd',eriAB[numpy.ix_(ivira,ivirb,ivira,ivirb)],sfva,dsvb,sfva,sfvb)\
              + einsum('abcd,a,b,c,d->abcd',eriAB[numpy.ix_(ivira,ivirb,ivira,ivirb)],sfva,sfvb,dsva,sfvb)\
              + einsum('abcd,a,b,c,d->abcd',eriAB[numpy.ix_(ivira,ivirb,ivira,ivirb)],sfva,sfvb,sfva,dsvb)

        Ivvvo = einsum('abci,a,b,c,i->abci',eriAB[numpy.ix_(ivira,ivirb,ivira,ioccb)],dsva,sfvb,sfva,sfob)\
              + einsum('abci,a,b,c,i->abci',eriAB[numpy.ix_(ivira,ivirb,ivira,ioccb)],sfva,dsvb,sfva,sfob)\
              + einsum('abci,a,b,c,i->abci',eriAB[numpy.ix_(ivira,ivirb,ivira,ioccb)],sfva,sfvb,dsva,sfob)\
              + einsum('abci,a,b,c,i->abci',eriAB[numpy.ix_(ivira,ivirb,ivira,ioccb)],sfva,sfvb,sfva,dsob)

        Ivvov = einsum('abic,a,b,i,c->abic',eriAB[numpy.ix_(ivira,ivirb,iocca,ivirb)],dsva,sfvb,sfoa,sfvb)\
              + einsum('abic,a,b,i,c->abic',eriAB[numpy.ix_(ivira,ivirb,iocca,ivirb)],sfva,dsvb,sfoa,sfvb)\
              + einsum('abic,a,b,i,c->abic',eriAB[numpy.ix_(ivira,ivirb,iocca,ivirb)],sfva,sfvb,sfoa,dsvb)\
              + einsum('abic,a,b,i,c->abic',eriAB[numpy.ix_(ivira,ivirb,iocca,ivirb)],sfva,sfvb,dsoa,sfvb)

        Ivovv = einsum('aibc,a,i,b,c->aibc',eriAB[numpy.ix_(ivira,ioccb,ivira,ivirb)],dsva,sfob,sfva,sfvb)\
              + einsum('aibc,a,i,b,c->aibc',eriAB[numpy.ix_(ivira,ioccb,ivira,ivirb)],sfva,dsob,sfva,sfvb)\
              + einsum('aibc,a,i,b,c->aibc',eriAB[numpy.ix_(ivira,ioccb,ivira,ivirb)],sfva,sfob,dsva,sfvb)\
              + einsum('aibc,a,i,b,c->aibc',eriAB[numpy.ix_(ivira,ioccb,ivira,ivirb)],sfva,sfob,sfva,dsvb)

        Iovvv = einsum('iabc,i,a,b,c->iabc',eriAB[numpy.ix_(iocca,ivirb,ivira,ivirb)],sfoa,dsvb,sfva,sfvb)\
              + einsum('iabc,i,a,b,c->iabc',eriAB[numpy.ix_(iocca,ivirb,ivira,ivirb)],dsoa,sfvb,sfva,sfvb)\
              + einsum('iabc,i,a,b,c->iabc',eriAB[numpy.ix_(iocca,ivirb,ivira,ivirb)],sfoa,sfvb,dsva,sfvb)\
              + einsum('iabc,i,a,b,c->iabc',eriAB[numpy.ix_(iocca,ivirb,ivira,ivirb)],sfoa,sfvb,sfva,dsvb)

        Ivvoo = einsum('abij,a,b,i,j->abij',eriAB[numpy.ix_(ivira,ivirb,iocca,ioccb)],dsva,sfvb,sfoa,sfob)\
              + einsum('abij,a,b,i,j->abij',eriAB[numpy.ix_(ivira,ivirb,iocca,ioccb)],sfva,dsvb,sfoa,sfob)\
              + einsum('abij,a,b,i,j->abij',eriAB[numpy.ix_(ivira,ivirb,iocca,ioccb)],sfva,sfvb,dsoa,sfob)\
              + einsum('abij,a,b,i,j->abij',eriAB[numpy.ix_(ivira,ivirb,iocca,ioccb)],sfva,sfvb,sfoa,dsob)

        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriAB[numpy.ix_(ivira,ioccb,ivira,ioccb)],dsva,sfob,sfva,sfob)\
              + einsum('ajbi,a,j,b,i->ajbi',eriAB[numpy.ix_(ivira,ioccb,ivira,ioccb)],sfva,dsob,sfva,sfob)\
              + einsum('ajbi,a,j,b,i->ajbi',eriAB[numpy.ix_(ivira,ioccb,ivira,ioccb)],sfva,sfob,dsva,sfob)\
              + einsum('ajbi,a,j,b,i->ajbi',eriAB[numpy.ix_(ivira,ioccb,ivira,ioccb)],sfva,sfob,sfva,dsob)

        Ioovv = einsum('ijab,i,j,a,b->ijab',eriAB[numpy.ix_(iocca,ioccb,ivira,ivirb)],dsoa,sfob,sfva,sfvb)\
              + einsum('ijab,i,j,a,b->ijab',eriAB[numpy.ix_(iocca,ioccb,ivira,ivirb)],sfoa,dsob,sfva,sfvb)\
              + einsum('ijab,i,j,a,b->ijab',eriAB[numpy.ix_(iocca,ioccb,ivira,ivirb)],sfoa,sfob,dsva,sfvb)\
              + einsum('ijab,i,j,a,b->ijab',eriAB[numpy.ix_(iocca,ioccb,ivira,ivirb)],sfoa,sfob,sfva,dsvb)

        Iovvo = einsum('jabi,j,a,b,i->jabi',eriAB[numpy.ix_(iocca,ivirb,ivira,ioccb)],sfoa,dsvb,sfva,sfob)\
              + einsum('jabi,j,a,b,i->jabi',eriAB[numpy.ix_(iocca,ivirb,ivira,ioccb)],dsoa,sfvb,sfva,sfob)\
              + einsum('jabi,j,a,b,i->jabi',eriAB[numpy.ix_(iocca,ivirb,ivira,ioccb)],sfoa,sfvb,dsva,sfob)\
              + einsum('jabi,j,a,b,i->jabi',eriAB[numpy.ix_(iocca,ivirb,ivira,ioccb)],sfoa,sfvb,sfva,dsob)

        Ivoov = einsum('ajib,a,j,i,b->ajib',eriAB[numpy.ix_(ivira,ioccb,iocca,ivirb)],dsva,sfob,sfoa,sfvb)\
              + einsum('ajib,a,j,i,b->ajib',eriAB[numpy.ix_(ivira,ioccb,iocca,ivirb)],sfva,dsob,sfoa,sfvb)\
              + einsum('ajib,a,j,i,b->ajib',eriAB[numpy.ix_(ivira,ioccb,iocca,ivirb)],sfva,sfob,sfoa,dsvb)\
              + einsum('ajib,a,j,i,b->ajib',eriAB[numpy.ix_(ivira,ioccb,iocca,ivirb)],sfva,sfob,dsoa,sfvb)

        Iovov = einsum('jaib,j,a,i,b->jaib',eriAB[numpy.ix_(iocca,ivirb,iocca,ivirb)],sfoa,dsvb,sfoa,sfvb)\
              + einsum('jaib,j,a,i,b->jaib',eriAB[numpy.ix_(iocca,ivirb,iocca,ivirb)],dsoa,sfvb,sfoa,sfvb)\
              + einsum('jaib,j,a,i,b->jaib',eriAB[numpy.ix_(iocca,ivirb,iocca,ivirb)],sfoa,sfvb,sfoa,dsvb)\
              + einsum('jaib,j,a,i,b->jaib',eriAB[numpy.ix_(iocca,ivirb,iocca,ivirb)],sfoa,sfvb,dsoa,sfvb)

        Ivooo = einsum('akij,a,k,i,j->akij',eriAB[numpy.ix_(ivira,ioccb,iocca,ioccb)],dsva,sfob,sfoa,sfob)\
              + einsum('akij,a,k,i,j->akij',eriAB[numpy.ix_(ivira,ioccb,iocca,ioccb)],sfva,dsob,sfoa,sfob)\
              + einsum('akij,a,k,i,j->akij',eriAB[numpy.ix_(ivira,ioccb,iocca,ioccb)],sfva,sfob,dsoa,sfob)\
              + einsum('akij,a,k,i,j->akij',eriAB[numpy.ix_(ivira,ioccb,iocca,ioccb)],sfva,sfob,sfoa,dsob)
 
        Iovoo = einsum('kaij,k,a,i,j->kaij',eriAB[numpy.ix_(iocca,ivirb,iocca,ioccb)],sfoa,dsvb,sfoa,sfob)\
              + einsum('kaij,k,a,i,j->kaij',eriAB[numpy.ix_(iocca,ivirb,iocca,ioccb)],dsoa,sfvb,sfoa,sfob)\
              + einsum('kaij,k,a,i,j->kaij',eriAB[numpy.ix_(iocca,ivirb,iocca,ioccb)],sfoa,sfvb,dsoa,sfob)\
              + einsum('kaij,k,a,i,j->kaij',eriAB[numpy.ix_(iocca,ivirb,iocca,ioccb)],sfoa,sfvb,sfoa,dsob)

        Ioovo = einsum('jkai,j,k,a,i->jkai',eriAB[numpy.ix_(iocca,ioccb,ivira,ioccb)],dsoa,sfob,sfva,sfob)\
              + einsum('jkai,j,k,a,i->jkai',eriAB[numpy.ix_(iocca,ioccb,ivira,ioccb)],sfoa,dsob,sfva,sfob)\
              + einsum('jkai,j,k,a,i->jkai',eriAB[numpy.ix_(iocca,ioccb,ivira,ioccb)],sfoa,sfob,sfva,dsob)\
              + einsum('jkai,j,k,a,i->jkai',eriAB[numpy.ix_(iocca,ioccb,ivira,ioccb)],sfoa,sfob,dsva,sfob)

        Iooov = einsum('jkia,j,k,i,a->jkia',eriAB[numpy.ix_(iocca,ioccb,iocca,ivirb)],dsoa,sfob,sfoa,sfvb)\
              + einsum('jkia,j,k,i,a->jkia',eriAB[numpy.ix_(iocca,ioccb,iocca,ivirb)],sfoa,dsob,sfoa,sfvb)\
              + einsum('jkia,j,k,i,a->jkia',eriAB[numpy.ix_(iocca,ioccb,iocca,ivirb)],sfoa,sfob,dsoa,sfvb)\
              + einsum('jkia,j,k,i,a->jkia',eriAB[numpy.ix_(iocca,ioccb,iocca,ivirb)],sfoa,sfob,sfoa,dsvb)

        Ioooo = einsum('klij,k,l,i,j->klij',eriAB[numpy.ix_(iocca,ioccb,iocca,ioccb)],dsoa,sfob,sfoa,sfob)\
              + einsum('klij,k,l,i,j->klij',eriAB[numpy.ix_(iocca,ioccb,iocca,ioccb)],sfoa,dsob,sfoa,sfob)\
              + einsum('klij,k,l,i,j->klij',eriAB[numpy.ix_(iocca,ioccb,iocca,ioccb)],sfoa,sfob,dsoa,sfob)\
              + einsum('klij,k,l,i,j->klij',eriAB[numpy.ix_(iocca,ioccb,iocca,ioccb)],sfoa,sfob,sfoa,dsob)

        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)

        return Fa,Fb,Ia,Ib,Iabab

def g_n2rdm_full(beta, sfo, sfv, P2):
    n2rdm = (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0],sfv,sfv,sfv,sfv)
    n2rdm += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1],sfv,sfo,sfv,sfv)
    n2rdm -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1],sfv,sfo,sfv,sfv)
    n2rdm += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2],sfv,sfv,sfv,sfo)
    n2rdm -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2],sfv,sfv,sfv,sfo)
    n2rdm += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3],sfo,sfo,sfv,sfv)
    n2rdm += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4],sfv,sfo,sfv,sfo)
    n2rdm -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4],sfv,sfo,sfv,sfo)
    n2rdm -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4],sfv,sfo,sfv,sfo)
    n2rdm += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4],sfv,sfo,sfv,sfo)
    n2rdm += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5],sfv,sfv,sfo,sfo)
    n2rdm += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6],sfo,sfo,sfv,sfo)
    n2rdm -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6],sfo,sfo,sfv,sfo)
    n2rdm += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7],sfo,sfv,sfo,sfo)
    n2rdm -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7],sfo,sfv,sfo,sfo)
    n2rdm += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8],sfo,sfo,sfo,sfo)
    return n2rdm

def g_n2rdm_full_active(beta, n, iocc, ivir, sfo, sfv, P2):
    n2rdm = numpy.zeros((n,n,n,n), dtype=P2[0].dtype)
    n2rdm[numpy.ix_(ivir,ivir,ivir,ivir)] += \
       (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0],sfv,sfv,sfv,sfv)
    n2rdm[numpy.ix_(ivir,iocc,ivir,ivir)] += \
       (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1],sfv,sfo,sfv,sfv)
    n2rdm[numpy.ix_(iocc,ivir,ivir,ivir)] -= \
       (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1],sfv,sfo,sfv,sfv)
    n2rdm[numpy.ix_(ivir,ivir,ivir,iocc)] += \
       (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2],sfv,sfv,sfv,sfo)
    n2rdm[numpy.ix_(ivir,ivir,iocc,ivir)] -= \
       (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2],sfv,sfv,sfv,sfo)
    n2rdm[numpy.ix_(iocc,iocc,ivir,ivir)] += \
       (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3],sfo,sfo,sfv,sfv)
    n2rdm[numpy.ix_(ivir,iocc,ivir,iocc)] += \
       (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4],sfv,sfo,sfv,sfo)
    n2rdm[numpy.ix_(ivir,iocc,iocc,ivir)] -= \
       (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4],sfv,sfo,sfv,sfo)
    n2rdm[numpy.ix_(iocc,ivir,ivir,iocc)] -= \
       (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4],sfv,sfo,sfv,sfo)
    n2rdm[numpy.ix_(iocc,ivir,iocc,ivir)] += \
       (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4],sfv,sfo,sfv,sfo)
    n2rdm[numpy.ix_(ivir,ivir,iocc,iocc)] += \
       (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5],sfv,sfv,sfo,sfo)
    n2rdm[numpy.ix_(iocc,iocc,ivir,iocc)] += \
       (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6],sfo,sfo,sfv,sfo)
    n2rdm[numpy.ix_(iocc,iocc,iocc,ivir)] -= \
       (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6],sfo,sfo,sfv,sfo)
    n2rdm[numpy.ix_(iocc,ivir,iocc,iocc)] += \
       (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7],sfo,sfv,sfo,sfo)
    n2rdm[numpy.ix_(ivir,iocc,iocc,iocc)] -= \
       (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7],sfo,sfv,sfo,sfo)
    n2rdm[numpy.ix_(iocc,iocc,iocc,iocc)] += \
       (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8],sfo,sfo,sfo,sfo)
    return n2rdm

def u_n2rdm_full(beta, sfoa, sfva, sfob, sfvb, P2):
    na = sfoa.size
    nb = sfob.size
    P2aa = numpy.zeros((na,na,na,na), dtype=P2[0][0].dtype)
    P2aa += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0][0],sfva,sfva,sfva,sfva)
    P2aa += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][0],sfva,sfoa,sfva,sfva)
    P2aa -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1][0],sfva,sfoa,sfva,sfva)
    P2aa += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][0],sfva,sfva,sfva,sfoa)
    P2aa -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2][0],sfva,sfva,sfva,sfoa)
    P2aa += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3][0],sfoa,sfoa,sfva,sfva)
    P2aa += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5][0],sfva,sfva,sfoa,sfoa)
    P2aa += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][0],sfoa,sfoa,sfva,sfoa)
    P2aa -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6][0],sfoa,sfoa,sfva,sfoa)
    P2aa += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][0],sfoa,sfva,sfoa,sfoa)
    P2aa -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7][0],sfoa,sfva,sfoa,sfoa)
    P2aa += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8][0],sfoa,sfoa,sfoa,sfoa)

    P2bb = numpy.zeros((nb,nb,nb,nb), dtype=P2[0][1].dtype)
    P2bb += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0][1],sfvb,sfvb,sfvb,sfvb)
    P2bb += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][1],sfvb,sfob,sfvb,sfvb)
    P2bb -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1][1],sfvb,sfob,sfvb,sfvb)
    P2bb += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][1],sfvb,sfvb,sfvb,sfob)
    P2bb -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2][1],sfvb,sfvb,sfvb,sfob)
    P2bb += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3][1],sfob,sfob,sfvb,sfvb)
    P2bb += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5][1],sfvb,sfvb,sfob,sfob)
    P2bb += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][1],sfob,sfob,sfvb,sfob)
    P2bb -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6][1],sfob,sfob,sfvb,sfob)
    P2bb += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][1],sfob,sfvb,sfob,sfob)
    P2bb -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7][1],sfob,sfvb,sfob,sfob)
    P2bb += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8][1],sfob,sfob,sfob,sfob)

    P2ab = numpy.zeros((na,nb,na,nb), dtype=P2[0][2].dtype)
    P2ab += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0][2],sfva,sfvb,sfva,sfvb)
    P2ab += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][2],sfva,sfob,sfva,sfvb)
    P2ab += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][2],sfva,sfvb,sfva,sfob)
    P2ab += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3][2],sfoa,sfob,sfva,sfvb)
    P2ab += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][2],sfva,sfob,sfva,sfob)
    P2ab += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5][2],sfva,sfvb,sfoa,sfob)
    P2ab += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][2],sfoa,sfob,sfva,sfob)
    P2ab += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][2],sfoa,sfvb,sfoa,sfob)
    P2ab += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8][2],sfoa,sfob,sfoa,sfob)

    P2ab += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][3],sfvb,sfoa,sfvb,sfva).transpose((1,0,3,2))
    P2ab += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][3],sfvb,sfva,sfvb,sfoa).transpose((1,0,3,2))

    P2ab += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][3],sfob,sfoa,sfvb,sfoa).transpose((1,0,3,2))
    P2ab += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][3],sfob,sfva,sfob,sfoa).transpose((1,0,3,2))

    P2ab -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][3],sfva,sfob,sfvb,sfoa).transpose((0,1,3,2))
    P2ab -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][4],sfvb,sfoa,sfva,sfob).transpose((1,0,2,3))
    P2ab += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][5],sfvb,sfoa,sfvb,sfoa).transpose((1,0,3,2))
    return (P2aa, P2bb, P2ab)

def u_n2rdm_full_active(beta, na, nb, iocca, ivira, ioccb, ivirb, sfoa, sfva, sfob, sfvb, P2):
    P2aa = numpy.zeros((na,na,na,na), dtype=P2[0][0].dtype)
    P2aa[numpy.ix_(ivira,ivira,ivira,ivira)] += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0][0],sfva,sfva,sfva,sfva)
    P2aa[numpy.ix_(ivira,iocca,ivira,ivira)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][0],sfva,sfoa,sfva,sfva)
    P2aa[numpy.ix_(iocca,ivira,ivira,ivira)] -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1][0],sfva,sfoa,sfva,sfva)
    P2aa[numpy.ix_(ivira,ivira,ivira,iocca)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][0],sfva,sfva,sfva,sfoa)
    P2aa[numpy.ix_(ivira,ivira,iocca,ivira)] -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2][0],sfva,sfva,sfva,sfoa)
    P2aa[numpy.ix_(iocca,iocca,ivira,ivira)] += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3][0],sfoa,sfoa,sfva,sfva)
    P2aa[numpy.ix_(ivira,iocca,ivira,iocca)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa[numpy.ix_(ivira,iocca,iocca,ivira)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa[numpy.ix_(iocca,ivira,ivira,iocca)] -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa[numpy.ix_(iocca,ivira,iocca,ivira)] += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4][0],sfva,sfoa,sfva,sfoa)
    P2aa[numpy.ix_(ivira,ivira,iocca,iocca)] += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5][0],sfva,sfva,sfoa,sfoa)
    P2aa[numpy.ix_(iocca,iocca,ivira,iocca)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][0],sfoa,sfoa,sfva,sfoa)
    P2aa[numpy.ix_(iocca,iocca,iocca,ivira)] -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6][0],sfoa,sfoa,sfva,sfoa)
    P2aa[numpy.ix_(iocca,ivira,iocca,iocca)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][0],sfoa,sfva,sfoa,sfoa)
    P2aa[numpy.ix_(ivira,iocca,iocca,iocca)] -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7][0],sfoa,sfva,sfoa,sfoa)
    P2aa[numpy.ix_(iocca,iocca,iocca,iocca)] += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8][0],sfoa,sfoa,sfoa,sfoa)

    P2bb = numpy.zeros((nb,nb,nb,nb), dtype=P2[0][1].dtype)
    P2bb[numpy.ix_(ivirb,ivirb,ivirb,ivirb)] += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0][1],sfvb,sfvb,sfvb,sfvb)
    P2bb[numpy.ix_(ivirb,ioccb,ivirb,ivirb)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][1],sfvb,sfob,sfvb,sfvb)
    P2bb[numpy.ix_(ioccb,ivirb,ivirb,ivirb)] -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1][1],sfvb,sfob,sfvb,sfvb)
    P2bb[numpy.ix_(ivirb,ivirb,ivirb,ioccb)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][1],sfvb,sfvb,sfvb,sfob)
    P2bb[numpy.ix_(ivirb,ivirb,ioccb,ivirb)] -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2][1],sfvb,sfvb,sfvb,sfob)
    P2bb[numpy.ix_(ioccb,ioccb,ivirb,ivirb)] += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3][1],sfob,sfob,sfvb,sfvb)
    P2bb[numpy.ix_(ivirb,ioccb,ivirb,ioccb)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb[numpy.ix_(ivirb,ioccb,ioccb,ivirb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb[numpy.ix_(ioccb,ivirb,ivirb,ioccb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb[numpy.ix_(ioccb,ivirb,ioccb,ivirb)] += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4][1],sfvb,sfob,sfvb,sfob)
    P2bb[numpy.ix_(ivirb,ivirb,ioccb,ioccb)] += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5][1],sfvb,sfvb,sfob,sfob)
    P2bb[numpy.ix_(ioccb,ioccb,ivirb,ioccb)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][1],sfob,sfob,sfvb,sfob)
    P2bb[numpy.ix_(ioccb,ioccb,ioccb,ivirb)] -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6][1],sfob,sfob,sfvb,sfob)
    P2bb[numpy.ix_(ioccb,ivirb,ioccb,ioccb)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][1],sfob,sfvb,sfob,sfob)
    P2bb[numpy.ix_(ivirb,ioccb,ioccb,ioccb)] -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7][1],sfob,sfvb,sfob,sfob)
    P2bb[numpy.ix_(ioccb,ioccb,ioccb,ioccb)] += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8][1],sfob,sfob,sfob,sfob)

    P2ab = numpy.zeros((na,nb,na,nb), dtype=P2[0][2].dtype)
    P2ab[numpy.ix_(ivira,ivirb,ivira,ivirb)] += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0][2],sfva,sfvb,sfva,sfvb)
    P2ab[numpy.ix_(ivira,ioccb,ivira,ivirb)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][2],sfva,sfob,sfva,sfvb)
    P2ab[numpy.ix_(ivira,ivirb,ivira,ioccb)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][2],sfva,sfvb,sfva,sfob)
    P2ab[numpy.ix_(iocca,ioccb,ivira,ivirb)] += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3][2],sfoa,sfob,sfva,sfvb)
    P2ab[numpy.ix_(ivira,ioccb,ivira,ioccb)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][2],sfva,sfob,sfva,sfob)
    P2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)] += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5][2],sfva,sfvb,sfoa,sfob)
    P2ab[numpy.ix_(iocca,ioccb,ivira,ioccb)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][2],sfoa,sfob,sfva,sfob)
    P2ab[numpy.ix_(iocca,ivirb,iocca,ioccb)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][2],sfoa,sfvb,sfoa,sfob)
    P2ab[numpy.ix_(iocca,ioccb,iocca,ioccb)] += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8][2],sfoa,sfob,sfoa,sfob)

    P2ab[numpy.ix_(iocca,ivirb,ivira,ivirb)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1][3],sfvb,sfoa,sfvb,sfva).transpose((1,0,3,2))
    P2ab[numpy.ix_(ivira,ivirb,iocca,ivirb)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2][3],sfvb,sfva,sfvb,sfoa).transpose((1,0,3,2))

    P2ab[numpy.ix_(iocca,ioccb,iocca,ivirb)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6][3],sfob,sfoa,sfvb,sfoa).transpose((1,0,3,2))
    P2ab[numpy.ix_(ivira,ioccb,iocca,iocca)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7][3],sfob,sfva,sfob,sfoa).transpose((1,0,3,2))

    P2ab[numpy.ix_(ivira,ioccb,iocca,ivirb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][3],sfva,sfob,sfvb,sfoa).transpose((0,1,3,2))
    P2ab[numpy.ix_(iocca,ivirb,ivira,ioccb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][4],sfvb,sfoa,sfva,sfob).transpose((1,0,2,3))
    P2ab[numpy.ix_(iocca,ivirb,iocca,ivirb)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4][5],sfvb,sfoa,sfvb,sfoa).transpose((1,0,3,2))
    return (P2aa, P2bb, P2ab)

def g_Fd_on(Fd, ndia, ndba, ndji, ndai):
    temp = -einsum('ia,aik->k',ndia,Fd)
    temp -= einsum('ba,abk->k',ndba,Fd)
    temp -= einsum('ji,ijk->k',ndji,Fd)
    temp -= einsum('ai,iak->k',ndai,Fd)
    return temp

def g_Fd_on_active(Fd, iocc, ivir, ndia, ndba, ndji, ndai):
    Fdai = Fd[numpy.ix_(ivir,iocc)]
    Fdab = Fd[numpy.ix_(ivir,ivir)]
    Fdij = Fd[numpy.ix_(iocc,iocc)]
    Fdia = Fd[numpy.ix_(iocc,ivir)]
    temp = -einsum('ia,aik->k',ndia,Fdai)
    temp -= einsum('ba,abk->k',ndba,Fdab)
    temp -= einsum('ji,ijk->k',ndji,Fdij)
    temp -= einsum('ai,iak->k',ndai,Fdia)
    return temp

def g_d_on_oo(dso, F, I, dia, dji, dai, P2, jitemp):
    jitemp -= 0.5*einsum('ia,ai->i',dia,F.vo)*dso
    jitemp -= 0.5*einsum('ji,ij->i',dji,F.oo)*dso
    jitemp -= 0.5*einsum('ji,ij->j',dji,F.oo)*dso
    jitemp -= 0.5*einsum('ai,ia->i',dai,F.ov)*dso

    jitemp -= 0.5*0.50*einsum('ciab,abci->i',P2[1],I.vvvo)*dso
    jitemp -= 0.5*0.50*einsum('bcai,aibc->i',P2[2],I.vovv)*dso
    jitemp -= 0.5*0.5*einsum('ijab,abij->i',P2[3],I.vvoo)*dso
    jitemp -= 0.5*1.00*einsum('bjai,aibj->i',P2[4],I.vovo)*dso
    jitemp -= 0.5*1.00*einsum('bjai,aibj->j',P2[4],I.vovo)*dso
    jitemp -= 0.5*0.5*einsum('abij,ijab->i',P2[5],I.oovv)*dso
    jitemp -= 0.5*0.50*einsum('jkai,aijk->i',P2[6],I.vooo)*dso
    jitemp -= 0.5*1.0*einsum('jkai,aijk->j',P2[6],I.vooo)*dso
    jitemp -= 0.5*1.0*einsum('kaij,ijka->i',P2[7],I.ooov)*dso
    jitemp -= 0.5*0.50*einsum('kaij,ijka->k',P2[7],I.ooov)*dso
    jitemp -= 0.5*0.5*einsum('klij,ijkl->i',P2[8],I.oooo)*dso
    jitemp -= 0.5*0.5*einsum('klij,ijkl->k',P2[8],I.oooo)*dso

def g_d_on_vv(dsv, F, I, dia, dba, dai, P2, batemp):
    batemp += 0.50*einsum('ia,ai->a',dia,F.vo)*dsv
    batemp += 0.50*einsum('ba,ab->b',dba,F.vv)*dsv
    batemp += 0.50*einsum('ba,ab->a',dba,F.vv)*dsv
    batemp += 0.50*einsum('ai,ia->a',dai,F.ov)*dsv

    batemp += 0.5*0.5*einsum('cdab,abcd->c',P2[0],I.vvvv)*dsv
    batemp += 0.5*0.5*einsum('cdab,abcd->a',P2[0],I.vvvv)*dsv
    batemp += 0.5*0.50*einsum('ciab,abci->c',P2[1],I.vvvo)*dsv
    batemp += 0.5*1.0*einsum('ciab,abci->a',P2[1],I.vvvo)*dsv
    batemp += 0.5*1.0*einsum('bcai,aibc->b',P2[2],I.vovv)*dsv
    batemp += 0.5*0.50*einsum('bcai,aibc->a',P2[2],I.vovv)*dsv
    batemp += 0.5*0.50*einsum('ijab,abij->a',P2[3],I.vvoo)*dsv
    batemp += 0.5*1.00*einsum('bjai,aibj->a',P2[4],I.vovo)*dsv
    batemp += 0.5*1.00*einsum('bjai,aibj->b',P2[4],I.vovo)*dsv
    batemp += 0.5*0.5*einsum('abij,ijab->a',P2[5],I.oovv)*dsv
    batemp += 0.5*0.50*einsum('jkai,aijk->a',P2[6],I.vooo)*dsv
    batemp += 0.5*0.50*einsum('kaij,ijka->a',P2[7],I.ooov)*dsv

def u_Fd_on(Fdaa, Fdab, Fdba, Fdbb, ndia, ndba, ndji, ndai):
    tempA = -einsum('ia,aik->k',ndia[0],Fdaa)
    tempA -= einsum('ba,abk->k',ndba[0],Fdaa)
    tempA -= einsum('ji,ijk->k',ndji[0],Fdaa)
    tempA -= einsum('ai,iak->k',ndai[0],Fdaa)
    tempA -= einsum('ia,aik->k',ndia[1],Fdba)
    tempA -= einsum('ba,abk->k',ndba[1],Fdba)
    tempA -= einsum('ji,ijk->k',ndji[1],Fdba)
    tempA -= einsum('ai,iak->k',ndai[1],Fdba)
    tempB = -einsum('ia,aik->k',ndia[1],Fdbb)
    tempB -= einsum('ba,abk->k',ndba[1],Fdbb)
    tempB -= einsum('ji,ijk->k',ndji[1],Fdbb)
    tempB -= einsum('ai,iak->k',ndai[1],Fdbb)
    tempB -= einsum('ia,aik->k',ndia[0],Fdab)
    tempB -= einsum('ba,abk->k',ndba[0],Fdab)
    tempB -= einsum('ji,ijk->k',ndji[0],Fdab)
    tempB -= einsum('ai,iak->k',ndai[0],Fdab)

    return tempA, tempB

def u_Fd_on_active(Fdaa, Fdab, Fdba, Fdbb, iocca, ivira, ioccb, ivirb, ndia, ndba, ndji, ndai):
    Fdaik = Fdaa[numpy.ix_(ivira,iocca)]
    Fdabk = Fdaa[numpy.ix_(ivira,ivira)]
    Fdijk = Fdaa[numpy.ix_(iocca,iocca)]
    Fdiak = Fdaa[numpy.ix_(iocca,ivira)]
    FdaiK = Fdab[numpy.ix_(ivira,iocca)]
    FdabK = Fdab[numpy.ix_(ivira,ivira)]
    FdijK = Fdab[numpy.ix_(iocca,iocca)]
    FdiaK = Fdab[numpy.ix_(iocca,ivira)]
    FdAIK = Fdbb[numpy.ix_(ivirb,ioccb)]
    FdABK = Fdbb[numpy.ix_(ivirb,ivirb)]
    FdIJK = Fdbb[numpy.ix_(ioccb,ioccb)]
    FdIAK = Fdbb[numpy.ix_(ioccb,ivirb)]
    FdAIk = Fdba[numpy.ix_(ivirb,ioccb)]
    FdABk = Fdba[numpy.ix_(ivirb,ivirb)]
    FdIJk = Fdba[numpy.ix_(ioccb,ioccb)]
    FdIAk = Fdba[numpy.ix_(ioccb,ivirb)]
    tempA = -einsum('ia,aik->k',ndia[0],Fdaik)
    tempA -= einsum('ba,abk->k',ndba[0],Fdabk)
    tempA -= einsum('ji,ijk->k',ndji[0],Fdijk)
    tempA -= einsum('ai,iak->k',ndai[0],Fdiak)
    tempA -= einsum('ia,aik->k',ndia[1],FdAIk)
    tempA -= einsum('ba,abk->k',ndba[1],FdABk)
    tempA -= einsum('ji,ijk->k',ndji[1],FdIJk)
    tempA -= einsum('ai,iak->k',ndai[1],FdIAk)
    tempB = -einsum('ia,aik->k',ndia[1],FdAIK)
    tempB -= einsum('ba,abk->k',ndba[1],FdABK)
    tempB -= einsum('ji,ijk->k',ndji[1],FdIJK)
    tempB -= einsum('ai,iak->k',ndai[1],FdIAK)
    tempB -= einsum('ia,aik->k',ndia[0],FdaiK)
    tempB -= einsum('ba,abk->k',ndba[0],FdabK)
    tempB -= einsum('ji,ijk->k',ndji[0],FdijK)
    tempB -= einsum('ai,iak->k',ndai[0],FdiaK)

    return tempA, tempB

def u_d_on_oo(dsoa, dsob, Fa, Fb, Ia, Ib, Iabab, dia, dji, dai, P2, jitempa, jitempb):
    jitempa -= 0.5*einsum('ia,ai->i', dia[0], Fa.vo)*dsoa
    jitempa -= 0.5*einsum('ji,ij->i', dji[0], Fa.oo)*dsoa
    jitempa -= 0.5*einsum('ji,ij->j', dji[0], Fa.oo)*dsoa
    jitempa -= 0.5*einsum('ai,ia->i', dai[0], Fa.ov)*dsoa

    jitempb -= 0.5*einsum('ia,ai->i', dia[1], Fb.vo)*dsob
    jitempb -= 0.5*einsum('ji,ij->i', dji[1], Fb.oo)*dsob
    jitempb -= 0.5*einsum('ji,ij->j', dji[1], Fb.oo)*dsob
    jitempb -= 0.5*einsum('ai,ia->i', dai[1], Fb.ov)*dsob

    jitempa -= 0.5*0.5*einsum('ijab,abij->i', P2[3][0], Ia.vvoo)*dsoa
    jitempb -= 0.5*0.5*einsum('ijab,abij->i', P2[3][1], Ib.vvoo)*dsob
    jitempa -= 0.5*1.0*einsum('iJaB,aBiJ->i', P2[3][2], Iabab.vvoo)*dsoa
    jitempb -= 0.5*1.0*einsum('iJaB,aBiJ->J', P2[3][2], Iabab.vvoo)*dsob

    jitempa -= 0.5*0.5*einsum('ciab,abci->i', P2[1][0], Ia.vvvo)*dsoa
    jitempb -= 0.5*0.5*einsum('ciab,abci->i', P2[1][1], Ib.vvvo)*dsob
    jitempb -= 0.5*1.0*einsum('ciab,abci->i', P2[1][2], Iabab.vvvo)*dsob
    jitempa -= 0.5*1.0*einsum('ciab,baic->i', P2[1][3], Iabab.vvov)*dsoa

    jitempa -= 0.5*0.5*einsum('jkai,aijk->i', P2[6][0], Ia.vooo)*dsoa
    jitempb -= 0.5*0.5*einsum('jkai,aijk->i', P2[6][1], Ib.vooo)*dsob
    jitempb -= 0.5*1.0*einsum('jKaI,aIjK->I', P2[6][2], Iabab.vooo)*dsob
    jitempa -= 0.5*1.0*einsum('JkAi,iAkJ->i', P2[6][3], Iabab.ovoo)*dsoa
    jitempa -= 0.5*1.0*einsum('jkai,aijk->j', P2[6][0], Ia.vooo)*dsoa
    jitempb -= 0.5*1.0*einsum('jkai,aijk->j', P2[6][1], Ib.vooo)*dsob
    jitempa -= 0.5*1.0*einsum('jKaI,aIjK->j', P2[6][2], Iabab.vooo)*dsoa
    jitempb -= 0.5*1.0*einsum('JkAi,iAkJ->J', P2[6][3], Iabab.ovoo)*dsob
    jitempb -= 0.5*1.0*einsum('jKaI,aIjK->K', P2[6][2], Iabab.vooo)*dsob
    jitempa -= 0.5*1.0*einsum('JkAi,iAkJ->k', P2[6][3], Iabab.ovoo)*dsoa

    jitempa -= 0.5*1.0*einsum('bjai,aibj->i', P2[4][0], Ia.vovo)*dsoa
    jitempa -= 0.5*1.0*einsum('bjai,aibj->j', P2[4][0], Ia.vovo)*dsoa
    jitempb -= 0.5*1.0*einsum('BJAI,AIBJ->I', P2[4][1], Ib.vovo)*dsob
    jitempb -= 0.5*1.0*einsum('BJAI,AIBJ->J', P2[4][1], Ib.vovo)*dsob
    jitempb -= 0.5*1.0*einsum('bJaI,aIbJ->I', P2[4][2], Iabab.vovo)*dsob
    jitempb -= 0.5*1.0*einsum('bJaI,aIbJ->J', P2[4][2], Iabab.vovo)*dsob
    jitempa += 0.5*1.0*einsum('bJAi,iAbJ->i', P2[4][3], Iabab.ovvo)*dsoa
    jitempb += 0.5*1.0*einsum('bJAi,iAbJ->J', P2[4][3], Iabab.ovvo)*dsob
    jitempb += 0.5*1.0*einsum('BjaI,aIjB->I', P2[4][4], Iabab.voov)*dsob
    jitempa += 0.5*1.0*einsum('BjaI,aIjB->j', P2[4][4], Iabab.voov)*dsoa
    jitempa -= 0.5*1.0*einsum('BjAi,iAjB->i', P2[4][5], Iabab.ovov)*dsoa
    jitempa -= 0.5*1.0*einsum('BjAi,iAjB->j', P2[4][5], Iabab.ovov)*dsoa

    jitempa -= 0.5*0.5*einsum('klij,ijkl->i', P2[8][0], Ia.oooo)*dsoa
    jitempa -= 0.5*0.5*einsum('klij,ijkl->k', P2[8][0], Ia.oooo)*dsoa
    jitempb -= 0.5*0.5*einsum('klij,ijkl->i', P2[8][1], Ib.oooo)*dsob
    jitempb -= 0.5*0.5*einsum('klij,ijkl->k', P2[8][1], Ib.oooo)*dsob
    jitempa -= 0.5*1.0*einsum('kLiJ,iJkL->i', P2[8][2], Iabab.oooo)*dsoa
    jitempb -= 0.5*1.0*einsum('kLiJ,iJkL->J', P2[8][2], Iabab.oooo)*dsob
    jitempa -= 0.5*1.0*einsum('kLiJ,iJkL->k', P2[8][2], Iabab.oooo)*dsoa
    jitempb -= 0.5*1.0*einsum('kLiJ,iJkL->L', P2[8][2], Iabab.oooo)*dsob

    jitempa -= 0.5*0.5*einsum('bcai,aibc->i', P2[2][0], Ia.vovv)*dsoa
    jitempb -= 0.5*0.5*einsum('bcai,aibc->i', P2[2][1], Ib.vovv)*dsob
    jitempb -= 0.5*1.0*einsum('bCaI,aIbC->I', P2[2][2], Iabab.vovv)*dsob
    jitempa -= 0.5*1.0*einsum('BcAi,iAcB->i', P2[2][3], Iabab.ovvv)*dsoa

    jitempa -= 0.5*1.0*einsum('kaij,ijka->i', P2[7][0], Ia.ooov)*dsoa
    jitempa -= 0.5*0.5*einsum('kaij,ijka->k', P2[7][0], Ia.ooov)*dsoa
    jitempb -= 0.5*1.0*einsum('kaij,ijka->i', P2[7][1], Ib.ooov)*dsob
    jitempb -= 0.5*0.5*einsum('kaij,ijka->k', P2[7][1], Ib.ooov)*dsob
    jitempa -= 0.5*1.0*einsum('kAiJ,iJkA->i', P2[7][2], Iabab.ooov)*dsoa
    jitempb -= 0.5*1.0*einsum('kAiJ,iJkA->J', P2[7][2], Iabab.ooov)*dsob
    jitempa -= 0.5*1.0*einsum('kAiJ,iJkA->k', P2[7][2], Iabab.ooov)*dsoa
    jitempb -= 0.5*1.0*einsum('KaIj,jIaK->I', P2[7][3], Iabab.oovo)*dsob
    jitempa -= 0.5*1.0*einsum('KaIj,jIaK->j', P2[7][3], Iabab.oovo)*dsoa
    jitempb -= 0.5*1.0*einsum('KaIj,jIaK->K', P2[7][3], Iabab.oovo)*dsob

    jitempa -= 0.5*0.5*einsum('abij,ijab->i', P2[5][0], Ia.oovv)*dsoa
    jitempb -= 0.5*0.5*einsum('abij,ijab->i', P2[5][1], Ib.oovv)*dsob
    jitempa -= 0.5*1.0*einsum('aBiJ,iJaB->i', P2[5][2], Iabab.oovv)*dsoa
    jitempb -= 0.5*1.0*einsum('aBiJ,iJaB->J', P2[5][2], Iabab.oovv)*dsob

def u_d_on_vv(dsva, dsvb, Fa, Fb, Ia, Ib, Iabab, dia, dba, dai, P2, batempa, batempb):
    batempa += 0.5*einsum('ia,ai->a', dia[0], Fa.vo)*dsva
    batempa += 0.5*einsum('ba,ab->a', dba[0], Fa.vv)*dsva
    batempa += 0.5*einsum('ba,ab->b', dba[0], Fa.vv)*dsva
    batempa += 0.5*einsum('ai,ia->a', dai[0], Fa.ov)*dsva

    batempb += 0.5*einsum('ia,ai->a', dia[1], Fb.vo)*dsvb
    batempb += 0.5*einsum('ba,ab->a', dba[1], Fb.vv)*dsvb
    batempb += 0.5*einsum('ba,ab->b', dba[1], Fb.vv)*dsvb
    batempb += 0.5*einsum('ai,ia->a', dai[1], Fb.ov)*dsvb

    batempa += 0.5*0.5*einsum('ijab,abij->a', P2[3][0], Ia.vvoo)*dsva
    batempb += 0.5*0.5*einsum('ijab,abij->a', P2[3][1], Ib.vvoo)*dsvb
    batempa += 0.5*1.0*einsum('iJaB,aBiJ->a', P2[3][2], Iabab.vvoo)*dsva
    batempb += 0.5*1.0*einsum('iJaB,aBiJ->B', P2[3][2], Iabab.vvoo)*dsvb

    batempa += 0.5*1.0*einsum('ciab,abci->a', P2[1][0], Ia.vvvo)*dsva
    batempa += 0.5*0.5*einsum('ciab,abci->c', P2[1][0], Ia.vvvo)*dsva
    batempb += 0.5*1.0*einsum('ciab,abci->a', P2[1][1], Ib.vvvo)*dsvb
    batempb += 0.5*0.5*einsum('ciab,abci->c', P2[1][1], Ib.vvvo)*dsvb
    batempa += 0.5*1.0*einsum('cIaB,aBcI->a', P2[1][2], Iabab.vvvo)*dsva
    batempb += 0.5*1.0*einsum('cIaB,aBcI->B', P2[1][2], Iabab.vvvo)*dsvb
    batempa += 0.5*1.0*einsum('cIaB,aBcI->c', P2[1][2], Iabab.vvvo)*dsva
    batempb += 0.5*1.0*einsum('CiAb,bAiC->A', P2[1][3], Iabab.vvov)*dsvb
    batempa += 0.5*1.0*einsum('CiAb,bAiC->b', P2[1][3], Iabab.vvov)*dsva
    batempb += 0.5*1.0*einsum('CiAb,bAiC->C', P2[1][3], Iabab.vvov)*dsvb

    batempa += 0.5*0.5*einsum('jkai,aijk->a', P2[6][0], Ia.vooo)*dsva
    batempb += 0.5*0.5*einsum('jkai,aijk->a', P2[6][1], Ib.vooo)*dsvb
    batempa += 0.5*1.0*einsum('jKaI,aIjK->a', P2[6][2], Iabab.vooo)*dsva
    batempb += 0.5*1.0*einsum('JkAi,iAkJ->A', P2[6][3], Iabab.ovoo)*dsvb

    batempa += 0.5*0.5*einsum('cdab,abcd->a', P2[0][0], Ia.vvvv)*dsva
    batempa += 0.5*0.5*einsum('cdab,abcd->c', P2[0][0], Ia.vvvv)*dsva
    batempb += 0.5*0.5*einsum('cdab,abcd->a', P2[0][1], Ib.vvvv)*dsvb
    batempb += 0.5*0.5*einsum('cdab,abcd->c', P2[0][1], Ib.vvvv)*dsvb
    batempa += 0.5*1.0*einsum('cDaB,aBcD->a', P2[0][2], Iabab.vvvv)*dsva
    batempb += 0.5*1.0*einsum('cDaB,aBcD->B', P2[0][2], Iabab.vvvv)*dsvb
    batempa += 0.5*1.0*einsum('cDaB,aBcD->c', P2[0][2], Iabab.vvvv)*dsva
    batempb += 0.5*1.0*einsum('cDaB,aBcD->D', P2[0][2], Iabab.vvvv)*dsvb

    batempa += 0.5*1.0*einsum('bjai,aibj->a', P2[4][0], Ia.vovo)*dsva
    batempa += 0.5*1.0*einsum('bjai,aibj->b', P2[4][0], Ia.vovo)*dsva
    batempb += 0.5*1.0*einsum('BJAI,AIBJ->A', P2[4][1], Ib.vovo)*dsvb
    batempb += 0.5*1.0*einsum('BJAI,AIBJ->B', P2[4][1], Ib.vovo)*dsvb
    batempa += 0.5*1.0*einsum('bJaI,aIbJ->a', P2[4][2], Iabab.vovo)*dsva
    batempa += 0.5*1.0*einsum('bJaI,aIbJ->b', P2[4][2], Iabab.vovo)*dsva
    batempb -= 0.5*1.0*einsum('bJAi,iAbJ->A', P2[4][3], Iabab.ovvo)*dsvb
    batempa -= 0.5*1.0*einsum('bJAi,iAbJ->b', P2[4][3], Iabab.ovvo)*dsva
    batempa -= 0.5*1.0*einsum('BjaI,aIjB->a', P2[4][4], Iabab.voov)*dsva
    batempb -= 0.5*1.0*einsum('BjaI,aIjB->B', P2[4][4], Iabab.voov)*dsvb
    batempb += 0.5*1.0*einsum('BjAi,iAjB->A', P2[4][5], Iabab.ovov)*dsvb
    batempb += 0.5*1.0*einsum('BjAi,iAjB->B', P2[4][5], Iabab.ovov)*dsvb

    batempa += 0.5*0.5*einsum('bcai,aibc->a', P2[2][0], Ia.vovv)*dsva
    batempa += 0.5*1.0*einsum('bcai,aibc->b', P2[2][0], Ia.vovv)*dsva
    batempb += 0.5*0.5*einsum('bcai,aibc->a', P2[2][1], Ib.vovv)*dsvb
    batempb += 0.5*1.0*einsum('bcai,aibc->b', P2[2][1], Ib.vovv)*dsvb
    batempa += 0.5*1.0*einsum('bCaI,aIbC->a', P2[2][2], Iabab.vovv)*dsva
    batempa += 0.5*1.0*einsum('bCaI,aIbC->b', P2[2][2], Iabab.vovv)*dsva
    batempb += 0.5*1.0*einsum('bCaI,aIbC->C', P2[2][2], Iabab.vovv)*dsvb
    batempb += 0.5*1.0*einsum('BcAi,iAcB->A', P2[2][3], Iabab.ovvv)*dsvb
    batempb += 0.5*1.0*einsum('BcAi,iAcB->B', P2[2][3], Iabab.ovvv)*dsvb
    batempa += 0.5*1.0*einsum('BcAi,iAcB->c', P2[2][3], Iabab.ovvv)*dsva

    batempa += 0.5*0.5*einsum('kaij,ijka->a', P2[7][0], Ia.ooov)*dsva
    batempb += 0.5*0.5*einsum('kaij,ijka->a', P2[7][1], Ib.ooov)*dsvb
    batempb += 0.5*1.0*einsum('kAiJ,iJkA->A', P2[7][2], Iabab.ooov)*dsvb
    batempa += 0.5*1.0*einsum('KaIj,jIaK->a', P2[7][3], Iabab.oovo)*dsva

    batempa += 0.5*0.5*einsum('abij,ijab->a', P2[5][0], Ia.oovv)*dsva
    batempb += 0.5*0.5*einsum('abij,ijab->a', P2[5][1], Ib.oovv)*dsvb
    batempa += 0.5*1.0*einsum('aBiJ,iJaB->a', P2[5][2], Iabab.oovv)*dsva
    batempb += 0.5*1.0*einsum('aBiJ,iJaB->B', P2[5][2], Iabab.oovv)*dsvb

def r_Fd_on(Fdss, Fdos, ndia, ndba, ndji, ndai):
    temp = -einsum('ia,aik->k',ndia,Fdss)
    temp -= einsum('ba,abk->k',ndba,Fdss)
    temp -= einsum('ji,ijk->k',ndji,Fdss)
    temp -= einsum('ai,iak->k',ndai,Fdss)
    temp -= einsum('ia,aik->k',ndia,Fdos)
    temp -= einsum('ba,abk->k',ndba,Fdos)
    temp -= einsum('ji,ijk->k',ndji,Fdos)
    temp -= einsum('ai,iak->k',ndai,Fdos)

    return temp

def r_Fd_on_active(Fdss, Fdos, iocc, ivir,ndia, ndba, ndji, ndai):
    Fdaik = Fdss[numpy.ix_(ivir,iocc)]
    Fdabk = Fdss[numpy.ix_(ivir,ivir)]
    Fdijk = Fdss[numpy.ix_(iocc,iocc)]
    Fdiak = Fdss[numpy.ix_(iocc,ivir)]
    FdaiK = Fdss[numpy.ix_(ivir,iocc)]
    FdabK = Fdss[numpy.ix_(ivir,ivir)]
    FdijK = Fdss[numpy.ix_(iocc,iocc)]
    FdiaK = Fdss[numpy.ix_(iocc,ivir)]
    FdAIK = Fdss[numpy.ix_(ivir,iocc)]
    FdABK = Fdss[numpy.ix_(ivir,ivir)]
    FdIJK = Fdss[numpy.ix_(iocc,iocc)]
    FdIAK = Fdss[numpy.ix_(iocc,ivir)]
    FdAIk = Fdos[numpy.ix_(ivir,iocc)]
    FdABk = Fdos[numpy.ix_(ivir,ivir)]
    FdIJk = Fdos[numpy.ix_(iocc,iocc)]
    FdIAk = Fdos[numpy.ix_(iocc,ivir)]
    temp = -einsum('ia,aik->k',ndia,Fdaik)
    temp -= einsum('ba,abk->k',ndba,Fdabk)
    temp -= einsum('ji,ijk->k',ndji,Fdijk)
    temp -= einsum('ai,iak->k',ndai,Fdiak)
    temp -= einsum('ia,aik->k',ndia,FdAIk)
    temp -= einsum('ba,abk->k',ndba,FdABk)
    temp -= einsum('ji,ijk->k',ndji,FdIJk)
    temp -= einsum('ai,iak->k',ndai,FdIAk)

    return temp

def r_d_on_oo(dso, F, I, dia, dji, dai, P2, jitemp):
    jitemp -= 0.5*einsum('ia,ai->i', dia, F.vo)*dso
    jitemp -= 0.5*einsum('ji,ij->i', dji, F.oo)*dso
    jitemp -= 0.5*einsum('ji,ij->j', dji, F.oo)*dso
    jitemp -= 0.5*einsum('ai,ia->i', dai, F.ov)*dso

    jitemp -= 0.5*0.5*einsum('ijab,abij->i', P2[3] - P2[3].transpose((0,1,3,2)), I.vvoo - I.vvoo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('iJaB,aBiJ->i', P2[3], I.vvoo)*dso

    jitemp -= 0.5*0.5*einsum('ciab,abci->i', P2[1] - P2[1].transpose((0,1,3,2)), I.vvvo - I.vvov.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('ciab,baic->i', P2[1], I.vvov)*dso

    jitemp -= 0.5*0.5*einsum('jkai,aijk->i', P2[7] - P2[7].transpose((1,0,2,3)), I.vooo - I.vooo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('JkAi,iAkJ->i', P2[7], I.ovoo)*dso
    jitemp -= 0.5*1.0*einsum('jkai,aijk->j', P2[7] - P2[7].transpose((1,0,2,3)), I.vooo - I.vooo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('jKaI,aIjK->j', P2[7], I.vooo)*dso
    jitemp -= 0.5*1.0*einsum('JkAi,iAkJ->k', P2[7], I.ovoo)*dso

    jitemp -= 0.5*1.0*einsum('bjai,aibj->i', P2[4] - P2[5].transpose((0,1,3,2)), I.vovo - I.voov.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('bjai,aibj->j', P2[4] - P2[5].transpose((0,1,3,2)), I.vovo - I.voov.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('bJAi,iAbJ->i', P2[5].transpose((0,1,3,2)), I.ovvo)*dso
    jitemp -= 0.5*1.0*einsum('BjaI,aIjB->j', P2[5].transpose((0,1,3,2)), I.voov)*dso
    jitemp -= 0.5*1.0*einsum('BjAi,iAjB->i', P2[4], I.ovov)*dso
    jitemp -= 0.5*1.0*einsum('BjAi,iAjB->j', P2[4], I.ovov)*dso

    jitemp -= 0.5*0.5*einsum('klij,ijkl->i', P2[9] - P2[9].transpose((0,1,3,2)), I.oooo - I.oooo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*0.5*einsum('klij,ijkl->k', P2[9] - P2[9].transpose((0,1,3,2)), I.oooo - I.oooo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('kLiJ,iJkL->i', P2[9], I.oooo)*dso
    jitemp -= 0.5*1.0*einsum('kLiJ,iJkL->k', P2[9], I.oooo)*dso

    jitemp -= 0.5*0.5*einsum('bcai,aibc->i', P2[2] - P2[2].transpose((1,0,2,3)), I.vovv - I.vovv.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('BcAi,iAcB->i', P2[2], I.ovvv)*dso

    jitemp -= 0.5*1.0*einsum('kaij,ijka->i', P2[8] - P2[8].transpose((0,1,3,2)), I.ooov - I.oovo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*0.5*einsum('kaij,ijka->k', P2[8] - P2[8].transpose((0,1,3,2)), I.ooov - I.oovo.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('kAiJ,iJkA->i', P2[8], I.ooov)*dso
    jitemp -= 0.5*1.0*einsum('kAiJ,iJkA->k', P2[8], I.ooov)*dso
    jitemp -= 0.5*1.0*einsum('KaIj,jIaK->j', P2[8], I.oovo)*dso

    jitemp -= 0.5*0.5*einsum('abij,ijab->i', P2[6] - P2[6].transpose((1,0,2,3)), I.oovv - I.oovv.transpose((0,1,3,2)))*dso
    jitemp -= 0.5*1.0*einsum('aBiJ,iJaB->i', P2[6], I.oovv)*dso

def r_d_on_vv(dsv, F, I, dia, dba, dai, P2, batemp):
    batemp += 0.5*einsum('ia,ai->a', dia, F.vo)*dsv
    batemp += 0.5*einsum('ba,ab->a', dba, F.vv)*dsv
    batemp += 0.5*einsum('ba,ab->b', dba, F.vv)*dsv
    batemp += 0.5*einsum('ai,ia->a', dai, F.ov)*dsv

    batemp += 0.5*0.5*einsum('ijab,abij->a', P2[3] - P2[3].transpose((0,1,3,2)), I.vvoo - I.vvoo.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('iJaB,aBiJ->a', P2[3], I.vvoo)*dsv

    batemp += 0.5*1.0*einsum('ciab,abci->a', P2[1] - P2[1].transpose((0,1,3,2)), I.vvvo - I.vvov.transpose((0,1,3,2)))*dsv
    batemp += 0.5*0.5*einsum('ciab,abci->c', P2[1] - P2[1].transpose((0,1,3,2)), I.vvvo - I.vvov.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('cIaB,aBcI->a', P2[1], I.vvvo)*dsv
    batemp += 0.5*1.0*einsum('cIaB,aBcI->c', P2[1], I.vvvo)*dsv
    batemp += 0.5*1.0*einsum('CiAb,bAiC->b', P2[1], I.vvov)*dsv

    batemp += 0.5*0.5*einsum('jkai,aijk->a', P2[7] - P2[7].transpose((1,0,2,3)), I.vooo - I.vooo.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('jKaI,aIjK->a', P2[7], I.vooo)*dsv

    batemp += 0.5*0.5*einsum('cdab,abcd->a', P2[0] - P2[0].transpose((0,1,3,2)), I.vvvv - I.vvvv.transpose((0,1,3,2)))*dsv
    batemp += 0.5*0.5*einsum('cdab,abcd->c', P2[0] - P2[0].transpose((0,1,3,2)), I.vvvv - I.vvvv.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('cDaB,aBcD->a', P2[0], I.vvvv)*dsv
    batemp += 0.5*1.0*einsum('cDaB,aBcD->c', P2[0], I.vvvv)*dsv

    batemp += 0.5*1.0*einsum('bjai,aibj->a', P2[4] - P2[5].transpose((0,1,3,2)), I.vovo - I.voov.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('bjai,aibj->b', P2[4] - P2[5].transpose((0,1,3,2)), I.vovo - I.voov.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('bJaI,aIbJ->a', P2[4], I.vovo)*dsv
    batemp += 0.5*1.0*einsum('bJaI,aIbJ->b', P2[4], I.vovo)*dsv
    batemp += 0.5*1.0*einsum('bJAi,iAbJ->b', P2[5].transpose((0,1,3,2)), I.ovvo)*dsv
    batemp += 0.5*1.0*einsum('BjaI,aIjB->a', P2[5].transpose((0,1,3,2)), I.voov)*dsv

    batemp += 0.5*0.5*einsum('bcai,aibc->a', P2[2] - P2[2].transpose((1,0,2,3)), I.vovv - I.vovv.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('bcai,aibc->b', P2[2] - P2[2].transpose((1,0,2,3)), I.vovv - I.vovv.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('bCaI,aIbC->a', P2[2], I.vovv)*dsv
    batemp += 0.5*1.0*einsum('bCaI,aIbC->b', P2[2], I.vovv)*dsv
    batemp += 0.5*1.0*einsum('BcAi,iAcB->c', P2[2], I.ovvv)*dsv

    batemp += 0.5*0.5*einsum('kaij,ijka->a', P2[8] - P2[8].transpose((0,1,3,2)), I.ooov - I.oovo.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('KaIj,jIaK->a', P2[8], I.oovo)*dsv

    batemp += 0.5*0.5*einsum('abij,ijab->a', P2[6] - P2[6].transpose((0,1,3,2)), I.oovv - I.oovv.transpose((0,1,3,2)))*dsv
    batemp += 0.5*1.0*einsum('aBiJ,iJaB->a', P2[6], I.oovv)*dsv

def g_full_rdm2(fo, n1rdm, rdm2):
    rdm2 += einsum('pr,qs->pqrs',numpy.diag(fo),numpy.diag(fo))
    rdm2 -= einsum('pr,qs->pqsr',numpy.diag(fo),numpy.diag(fo))
    rdm2 += 0.5*einsum('pr,qs->pqrs',numpy.diag(fo),n1rdm)
    rdm2 -= 0.5*einsum('pr,qs->pqsr',numpy.diag(fo),n1rdm)
    rdm2 += 0.5*einsum('pr,qs->pqrs',n1rdm,numpy.diag(fo))
    rdm2 -= 0.5*einsum('pr,qs->pqsr',n1rdm,numpy.diag(fo))

def g_full_rdm2_active(focc, iocc, iall, n1rdm, rdm2):
    rdm2[numpy.ix_(iocc,iocc,iocc,iocc)] += einsum('pr,qs->pqrs',numpy.diag(focc),numpy.diag(focc))
    rdm2[numpy.ix_(iocc,iocc,iocc,iocc)] -= einsum('pr,qs->pqsr',numpy.diag(focc),numpy.diag(focc))
    rdm2[numpy.ix_(iocc,iall,iocc,iall)] += 0.5*einsum('pr,qs->pqrs',numpy.diag(focc),n1rdm)
    rdm2[numpy.ix_(iocc,iall,iall,iocc)] -= 0.5*einsum('pr,qs->pqsr',numpy.diag(focc),n1rdm)
    rdm2[numpy.ix_(iall,iocc,iall,iocc)] += 0.5*einsum('pr,qs->pqrs',n1rdm,numpy.diag(focc))
    rdm2[numpy.ix_(iall,iocc,iocc,iall)] -= 0.5*einsum('pr,qs->pqsr',n1rdm,numpy.diag(focc))

def u_full_rdm2(foa, fob, n1rdm, rdm2):
    rdm2[0] += einsum('pr,qs->pqrs',numpy.diag(foa),numpy.diag(foa))
    rdm2[0] -= einsum('pr,qs->pqsr',numpy.diag(foa),numpy.diag(foa))
    rdm2[0] += 0.5*einsum('pr,qs->pqrs',numpy.diag(foa),n1rdm[0])
    rdm2[0] -= 0.5*einsum('pr,qs->pqsr',numpy.diag(foa),n1rdm[0])
    rdm2[0] += 0.5*einsum('pr,qs->pqrs',n1rdm[0],numpy.diag(foa))
    rdm2[0] -= 0.5*einsum('pr,qs->pqsr',n1rdm[0],numpy.diag(foa))

    rdm2[1] += einsum('pr,qs->pqrs',numpy.diag(fob),numpy.diag(fob))
    rdm2[1] -= einsum('pr,qs->pqsr',numpy.diag(fob),numpy.diag(foa))
    rdm2[1] += 0.5*einsum('pr,qs->pqrs',numpy.diag(fob),n1rdm[1])
    rdm2[1] -= 0.5*einsum('pr,qs->pqsr',numpy.diag(fob),n1rdm[1])
    rdm2[1] += 0.5*einsum('pr,qs->pqrs',n1rdm[1],numpy.diag(fob))
    rdm2[1] -= 0.5*einsum('pr,qs->pqsr',n1rdm[1],numpy.diag(fob))

    rdm2[2] += einsum('pr,qs->pqrs',numpy.diag(foa),numpy.diag(fob))
    rdm2[2] += 0.5*einsum('pr,qs->pqrs',numpy.diag(foa),n1rdm[1])
    rdm2[2] += 0.5*einsum('pr,qs->pqrs',n1rdm[0],numpy.diag(fob))

def u_full_rdm2_active(focca, foccb, iocca, ioccb, ialla, iallb, n1rdm, rdm2):
    rdm2[0][numpy.ix_(iocca,iocca,iocca,iocca)] += einsum('pr,qs->pqrs',numpy.diag(focca),numpy.diag(focca))
    rdm2[0][numpy.ix_(iocca,iocca,iocca,iocca)] -= einsum('pr,qs->pqsr',numpy.diag(focca),numpy.diag(focca))
    rdm2[0][numpy.ix_(iocca,ialla,iocca,ialla)] += 0.5*einsum('pr,qs->pqrs',numpy.diag(focca),n1rdm[0])
    rdm2[0][numpy.ix_(iocca,ialla,ialla,iocca)] -= 0.5*einsum('pr,qs->pqsr',numpy.diag(focca),n1rdm[0])
    rdm2[0][numpy.ix_(ialla,iocca,ialla,iocca)] += 0.5*einsum('pr,qs->pqrs',n1rdm[0],numpy.diag(focca))
    rdm2[0][numpy.ix_(ialla,iocca,iocca,ialla)] -= 0.5*einsum('pr,qs->pqsr',n1rdm[0],numpy.diag(focca))

    rdm2[1][numpy.ix_(ioccb,ioccb,ioccb,ioccb)] += einsum('pr,qs->pqrs',numpy.diag(foccb),numpy.diag(foccb))
    rdm2[1][numpy.ix_(ioccb,ioccb,ioccb,ioccb)] -= einsum('pr,qs->pqsr',numpy.diag(foccb),numpy.diag(foccb))
    rdm2[1][numpy.ix_(ioccb,iallb,ioccb,iallb)] += 0.5*einsum('pr,qs->pqrs',numpy.diag(foccb),n1rdm[1])
    rdm2[1][numpy.ix_(ioccb,iallb,iallb,ioccb)] -= 0.5*einsum('pr,qs->pqsr',numpy.diag(foccb),n1rdm[1])
    rdm2[1][numpy.ix_(iallb,ioccb,iallb,ioccb)] += 0.5*einsum('pr,qs->pqrs',n1rdm[1],numpy.diag(foccb))
    rdm2[1][numpy.ix_(iallb,ioccb,ioccb,iallb)] -= 0.5*einsum('pr,qs->pqsr',n1rdm[1],numpy.diag(foccb))

    rdm2[2][numpy.ix_(iocca,ioccb,iocca,ioccb)] += einsum('pr,qs->pqrs',numpy.diag(focca),numpy.diag(foccb))
    rdm2[2][numpy.ix_(iocca,iallb,iocca,iallb)] += 0.5*einsum('pr,qs->pqrs',numpy.diag(focca),n1rdm[1])
    rdm2[2][numpy.ix_(ialla,ioccb,ialla,ioccb)] += 0.5*einsum('pr,qs->pqrs',n1rdm[0],numpy.diag(foccb))
