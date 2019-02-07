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
    nl1 = numpy.linalg.norm(T1old) + 0.0001
    nl2 = numpy.linalg.norm(T2old) + 0.0001
    while i < max_iter and not converged:
        # form new T1 and T2
        T1,T2 = form_new_ampl(method,F,I,T1old,T2old,D1,D2,ti,ng,G)

        res1 = numpy.linalg.norm(T1 - T1old) / nl1
        res2 = numpy.linalg.norm(T2 - T2old) / nl2
        # damp new T-amplitudes
        T1old = alpha*T1old + (1.0 - alpha)*T1
        T2old = alpha*T2old + (1.0 - alpha)*T2
        nl1 = numpy.linalg.norm(T1old) + 0.000001
        nl2 = numpy.linalg.norm(T2old) + 0.000001

        # compute energy
        E = ft_cc_energy.ft_cc_energy(T1old,T2old,
            F.ov,I.oovv,g,beta)

        # determine convergence
        if iprint > 0:
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

    return Eold,T1,T2

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
    t1bar = numpy.zeros((ng,nv,no))
    t2bar = numpy.zeros((ng,nv,nv,no,no))
    T1new = numpy.zeros((ng,nv,no))
    T2new = numpy.zeros((ng,nv,nv,no,no))

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

        nl1 = numpy.linalg.norm(T1aold) + 0.0001
        nl1 += numpy.linalg.norm(T1bold)
        nl2 = numpy.linalg.norm(T2aaold) + 0.0001
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
    nl1 = numpy.linalg.norm(L1old)
    nl2 = numpy.linalg.norm(L2old)
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
        nl1 = numpy.linalg.norm(L1old)
        nl2 = numpy.linalg.norm(L2old)
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

    nl1 = numpy.linalg.norm(L1aold) + numpy.linalg.norm(L1bold)
    nl2 = numpy.linalg.norm(L2aaold)
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
        nl1 = numpy.linalg.norm(L1aold) + numpy.linalg.norm(L1bold)
        nl2 = numpy.linalg.norm(L2aaold)
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

def get_ft_integrals(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get FT fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        # get ERIs
        eri = sys.g_aint_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,j->ij',fmo,fo)
        Fvo = einsum('ai,a,i->ai',fmo,fv,fo)
        Fvv = einsum('ab,a->ab',fmo,fv)
        F = one_e_blocks(Foo,fmo,Fvo,Fvv)

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
        return F,I

def get_ft_integrals_neq(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
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

def get_uft_integrals(sys, ea, eb, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        na = ea.shape[0]
        nb = eb.shape[0]
        en = numpy.concatenate((ea,eb))
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)

        # get FT fock matrix
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)

        # pre-contract with fermi factors
        Fooa = einsum('ij,j->ij',fa,foa)
        Fvoa = einsum('ai,a,i->ai',fa,fva,foa)
        Fvva = einsum('ab,a->ab',fa,fva)
        Fa = one_e_blocks(Fooa,fa,Fvoa,Fvva)

        Foob = einsum('ij,j->ij',fb,fob)
        Fvob = einsum('ai,a,i->ai',fb,fvb,fob)
        Fvvb = einsum('ab,a->ab',fb,fvb)
        Fb = one_e_blocks(Foob,fb,Fvob,Fvvb)

        # get ERIs
        eriA,eriB,eriAB = sys.u_aint_tot()
        Ivvvv = einsum('abcd,a,b->abcd',eriA,fva,fva)
        Ivvvo = einsum('abci,a,b,i->abci',eriA,fva,fva,foa)
        Ivovv = einsum('aibc,a->aibc',eriA,fva)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriA,fva,fva,foa,foa)
        Ivovo = einsum('ajbi,a,i->ajbi',eriA,fva,foa)
        Ivooo = einsum('akij,a,i,j->akij',eriA,fva,foa,foa)
        Iooov = einsum('jkia,i->jkia',eriA,foa)
        Ioooo = einsum('klij,i,j->klij',eriA,foa,foa)
        Ia = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=eriA,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b->abcd',eriB,fvb,fvb)
        Ivvvo = einsum('abci,a,b,i->abci',eriB,fvb,fvb,fob)
        Ivovv = einsum('aibc,a->aibc',eriB,fvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriB,fvb,fvb,fob,fob)
        Ivovo = einsum('ajbi,a,i->ajbi',eriB,fvb,fob)
        Ivooo = einsum('akij,a,i,j->akij',eriB,fvb,fob,fob)
        Iooov = einsum('jkia,i->jkia',eriB,fob)
        Ioooo = einsum('klij,i,j->klij',eriB,fob,fob)
        Ib = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=eriB,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b->abcd',eriAB,fva,fvb)
        Ivvvo = einsum('abci,a,b,i->abci',eriAB,fva,fvb,fob)
        Ivvov = einsum('abic,a,b,i->abic',eriAB,fva,fvb,foa)
        Ivovv = einsum('aibc,a->aibc',eriAB,fva)
        Iovvv = einsum('iabc,a->iabc',eriAB,fvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriAB,fva,fvb,foa,fob)
        Ivovo = einsum('ajbi,a,i->ajbi',eriAB,fva,fob)
        Iovvo = einsum('jabi,a,i->jabi',eriAB,fvb,fob)
        Ivoov = einsum('ajib,a,i->ajib',eriAB,fva,foa)
        Iovov = einsum('jaib,a,i->jaib',eriAB,fvb,foa)
        Ivooo = einsum('akij,a,i,j->akij',eriAB,fva,foa,fob)
        Iovoo = einsum('kaij,a,i,j->kaij',eriAB,fvb,foa,fob)
        Ioovo = einsum('jkai,i->jkai',eriAB,fob)
        Iooov = einsum('jkia,i->jkia',eriAB,foa)
        Ioooo = einsum('klij,i,j->klij',eriAB,foa,fob)
        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=eriAB,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)

        return Fa,Fb,Ia,Ib,Iabab

def get_ft_active_integrals(sys, en, focc, fvir, iocc, ivir):
        """Return one and two-electron integrals in the general spin orbital basis
        with different ."""
        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        # get ERIs
        eri = sys.g_aint_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,j->ij',fmo[numpy.ix_(iocc,iocc)],focc)
        Fvo = einsum('ai,a,i->ai',fmo[numpy.ix_(ivir,iocc)],fvir,focc)
        Fvv = einsum('ab,a->ab',fmo[numpy.ix_(ivir,ivir)],fvir)
        Fov = fmo[numpy.ix_(iocc,ivir)]
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],fvir,fvir)
        Ivvvo = einsum('abci,a,b,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],fvir,fvir,focc)
        Ivovv = einsum('aibc,a->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],fvir)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],fvir,fvir,focc,focc)
        Ioovv = eri[numpy.ix_(iocc,iocc,ivir,ivir)]
        Ivovo = einsum('ajbi,a,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],fvir,focc)
        Ivooo = einsum('akij,a,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],fvir,focc,focc)
        Iooov = einsum('jkia,i->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],focc)
        Ioooo = einsum('klij,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],focc,focc)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        return F,I

def get_ft_d_integrals(sys, en, fo, fv, dvec):
        """form integrals contracted with derivatives of occupation numbers.""" 

        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)
        fd = sys.g_fock_d_tot(dvec)

        # get ERIs
        eri = sys.g_aint_tot()

        # form derivative integrals
        fov = dvec*fo*fv
        Foo = einsum('ij,j->ij',fmo,fov) + einsum('ij,j->ij',fd,fo)
        Fvo = einsum('ai,a,i->ai',fd,fv,fo) + einsum('ai,a,i->ai',fmo,fv,fov) \
                - einsum('ai,a,i->ai',fmo,fov,fo)
        Fvv = einsum('ab,a->ab',fd,fv) - einsum('ab,a->ab',fmo,fov)
        F = one_e_blocks(Foo,fd,Fvo,Fvv)

        Ivvvv = - einsum('abcd,a,b->abcd',eri,fov,fv)\
                - einsum('abcd,a,b->abcd',eri,fv,fov)
        Ivvvo = - einsum('abci,a,b,i->abci',eri,fov,fv,fo)\
                - einsum('abci,a,b,i->abci',eri,fv,fov,fo)\
                + einsum('abci,a,b,i->abci',eri,fv,fv,fov)
        Ivovv = - einsum('aibc,a->aibc',eri,fov)
        Ivvoo = - einsum('abij,a,b,i,j->abij',eri,fov,fv,fo,fo)\
                - einsum('abij,a,b,i,j->abij',eri,fv,fov,fo,fo)\
                + einsum('abij,a,b,i,j->abij',eri,fv,fv,fov,fo)\
                + einsum('abij,a,b,i,j->abij',eri,fv,fv,fo,fov)
        Ivovo = - einsum('ajbi,a,i->ajbi',eri,fov,fo) \
                + einsum('ajbi,a,i->ajbi',eri,fv,fov)
        Ivooo = -einsum('akij,a,i,j->akij',eri,fov,fo,fo)\
                + einsum('akij,a,i,j->akij',eri,fv,fov,fo)\
                + einsum('akij,a,i,j->akij',eri,fv,fo,fov)
        Iooov = einsum('jkia,i->jkia',eri,fov)
        Ioooo = einsum('klij,i,j->klij',eri,fov,fo) \
                + einsum('klij,i,j->klij',eri,fo,fov)
        Ioovv = numpy.zeros(eri.shape)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return F,I

def u_ft_d_integrals(sys, ea, eb, foa, fob, fva, fvb, dveca, dvecb):
        """form integrals contracted with derivatives of occupation numbers."""
        na = ea.shape[0]
        nb = eb.shape[0]

        # get FT Fock matrices
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)
        fda,fdb = sys.u_fock_d_tot(dveca,dvecb)

        # form derivative integrals
        fova = dveca*foa*fva
        fovb = dvecb*fob*fvb

        Fooa = einsum('ij,j->ij',fa,fova) + einsum('ij,j->ij',fda,foa)
        Fvoa = einsum('ai,a,i->ai',fda,fva,foa) + einsum('ai,a,i->ai',fa,fva,fova) \
                - einsum('ai,a,i->ai',fa,fova,foa)
        Fvva = einsum('ab,a->ab',fda,fva) - einsum('ab,a->ab',fa,fova)
        Foob = einsum('ij,j->ij',fa,fova) + einsum('ij,j->ij',fda,foa)
        Fvob = einsum('ai,a,i->ai',fdb,fvb,fob) + einsum('ai,a,i->ai',fb,fvb,fovb) \
                - einsum('ai,a,i->ai',fb,fovb,fob)
        Fvvb = einsum('ab,a->ab',fdb,fvb) - einsum('ab,a->ab',fb,fovb)
        Fa = one_e_blocks(Fooa,fda,Fvoa,Fvva)
        Fb = one_e_blocks(Foob,fdb,Fvob,Fvvb)

        # get ERIs
        Ia,Ib,Iabab = sys.u_aint_tot()

        Iavvvv = - einsum('abcd,a,b->abcd',Ia,fova,fva)\
                - einsum('abcd,a,b->abcd',Ia,fva,fova)
        Iavvvo = - einsum('abci,a,b,i->abci',Ia,fova,fva,foa)\
                - einsum('abci,a,b,i->abci',Ia,fva,fova,foa)\
                + einsum('abci,a,b,i->abci',Ia,fva,fva,fova)
        Iavovv = - einsum('aibc,a->aibc',Ia,fova)
        Iavvoo = - einsum('abij,a,b,i,j->abij',Ia,fova,fva,foa,foa)\
                - einsum('abij,a,b,i,j->abij',Ia,fva,fova,foa,foa)\
                + einsum('abij,a,b,i,j->abij',Ia,fva,fva,fova,foa)\
                + einsum('abij,a,b,i,j->abij',Ia,fva,fva,foa,fova)
        Iavovo = - einsum('ajbi,a,i->ajbi',Ia,fova,foa) \
                + einsum('ajbi,a,i->ajbi',Ia,fva,fova)
        Iavooo = -einsum('akij,a,i,j->akij',Ia,fova,foa,foa)\
                + einsum('akij,a,i,j->akij',Ia,fva,fova,foa)\
                + einsum('akij,a,i,j->akij',Ia,fva,foa,fova)
        Iaooov = einsum('jkia,i->jkia',Ia,fova)
        Iaoooo = einsum('klij,i,j->klij',Ia,fova,foa) \
                + einsum('klij,i,j->klij',Ia,foa,fova)
        Iaoovv = numpy.zeros(Ia.shape)
        Ia = two_e_blocks(vvvv=Iavvvv,vvvo=Iavvvo,vovv=Iavovv,vvoo=Iavvoo,
                vovo=Iavovo,oovv=Iaoovv,vooo=Iavooo,ooov=Iaooov,oooo=Iaoooo)

        Ibvvvv = -einsum('abcd,a,b->abcd',Ib,fovb,fvb)\
                - einsum('abcd,a,b->abcd',Ib,fvb,fovb)
        Ibvvvo = -einsum('abci,a,b,i->abci',Ib,fovb,fvb,fob)\
                - einsum('abci,a,b,i->abci',Ib,fvb,fovb,fob)\
                + einsum('abci,a,b,i->abci',Ib,fvb,fvb,fovb)
        Ibvovv = -einsum('aibc,a->aibc',Ib,fovb)
        Ibvvoo = -einsum('abij,a,b,i,j->abij',Ib,fovb,fvb,fob,fob)\
                - einsum('abij,a,b,i,j->abij',Ib,fvb,fovb,fob,fob)\
                + einsum('abij,a,b,i,j->abij',Ib,fvb,fvb,fovb,fob)\
                + einsum('abij,a,b,i,j->abij',Ib,fvb,fvb,fob,fovb)
        Ibvovo = -einsum('ajbi,a,i->ajbi',Ib,fovb,fob) \
                + einsum('ajbi,a,i->ajbi',Ib,fvb,fovb)
        Ibvooo = -einsum('akij,a,i,j->akij',Ib,fovb,fob,fob)\
                + einsum('akij,a,i,j->akij',Ib,fvb,fovb,fob)\
                + einsum('akij,a,i,j->akij',Ib,fvb,fob,fovb)
        Ibooov = einsum('jkia,i->jkia',Ib,fovb)
        Iboooo = einsum('klij,i,j->klij',Ib,fovb,fob) \
                + einsum('klij,i,j->klij',Ib,fob,fovb)
        Iboovv = numpy.zeros(Ib.shape)
        Ib = two_e_blocks(vvvv=Ibvvvv,vvvo=Ibvvvo,vovv=Ibvovv,vvoo=Ibvvoo,
                vovo=Ibvovo,oovv=Iboovv,vooo=Ibvooo,ooov=Ibooov,oooo=Iboooo)

        I2vvvv = -einsum('abcd,a,b->abcd',Iabab,fova,fvb)\
                - einsum('abcd,a,b->abcd',Iabab,fva,fovb)
        I2vvvo = -einsum('abci,a,b,i->abci',Iabab,fova,fvb,fob)\
                - einsum('abci,a,b,i->abci',Iabab,fva,fovb,fob)\
                + einsum('abci,a,b,i->abci',Iabab,fva,fvb,fovb)
        I2vvov = -einsum('abic,a,b,i->abic',Iabab,fova,fvb,foa)\
                - einsum('abic,a,b,i->abic',Iabab,fva,fovb,foa)\
                + einsum('abic,a,b,i->abic',Iabab,fva,fvb,fova)
        I2vovv = -einsum('aibc,a->aibc',Iabab,fova)
        I2ovvv = -einsum('iabc,a->iabc',Iabab,fovb)
        I2vvoo = -einsum('abij,a,b,i,j->abij',Iabab,fova,fvb,foa,fob)\
                - einsum('abij,a,b,i,j->abij',Iabab,fva,fovb,foa,fob)\
                + einsum('abij,a,b,i,j->abij',Iabab,fva,fvb,fova,fob)\
                + einsum('abij,a,b,i,j->abij',Iabab,fva,fvb,foa,fovb)
        I2vovo = -einsum('ajbi,a,i->ajbi',Iabab,fova,fob) \
                + einsum('ajbi,a,i->ajbi',Iabab,fva,fovb)
        I2ovvo = -einsum('jabi,a,i->jabi',Iabab,fovb,fob) \
                + einsum('jabi,a,i->jabi',Iabab,fvb,fovb)
        I2voov = -einsum('ajib,a,i->ajib',Iabab,fova,foa) \
                + einsum('ajib,a,i->ajib',Iabab,fva,fova)
        I2ovov = -einsum('jaib,a,i->jaib',Iabab,fovb,foa) \
                + einsum('jaib,a,i->jaib',Iabab,fvb,fova)
        I2vooo = -einsum('akij,a,i,j->akij',Iabab,fova,foa,fob)\
                + einsum('akij,a,i,j->akij',Iabab,fva,fova,fob)\
                + einsum('akij,a,i,j->akij',Iabab,fva,foa,fovb)
        I2ovoo = -einsum('kaij,a,i,j->kaij',Iabab,fovb,foa,fob)\
                + einsum('kaij,a,i,j->kaij',Iabab,fvb,fova,fob)\
                + einsum('kaij,a,i,j->kaij',Iabab,fvb,foa,fovb)
        I2ooov = einsum('jkia,i->jkia',Iabab,fova)
        I2oovo = einsum('jkai,i->jkai',Iabab,fovb)
        I2oooo = einsum('klij,i,j->klij',Iabab,fova,fob) \
                + einsum('klij,i,j->klij',Iabab,foa,fovb)
        I2oovv = numpy.zeros(Iabab.shape)
        Iabab = two_e_blocks_full(vvvv=I2vvvv,
                vvvo=I2vvvo,vvov=I2vvov,
                vovv=I2vovv,ovvv=I2ovvv,
                vvoo=I2vvoo,vovo=I2vovo,
                ovvo=I2ovvo,voov=I2voov,
                ovov=I2ovov,oovv=I2oovv,
                vooo=I2vooo,ovoo=I2ovoo,
                oovo=I2oovo,ooov=I2ooov,
                oooo=I2oooo)


        return Fa,Fb,Ia,Ib,Iabab
