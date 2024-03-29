import numpy
from pyscf import lib
from cqcpy.ov_blocks import one_e_blocks
from cqcpy import cc_equations
from . import quadrature

einsum = lib.einsum
#einsum = einsum


def lccd_simple(F, I, T2old, D2, ti, ng, G):
    """Time-dependent linearized coupled cluster
    doubles (LCCD) iteration.
    """
    Id = numpy.ones((ng))
    T2new = -einsum('v,abij->vabij', Id, I.vvoo)

    for y in range(ng):
        cc_equations._D_D(T2new[y], F, I, T2old[y], fac=-1.0)

    # integrate the new T-amplitudes
    T2new = quadrature.int_tbar2(ng, T2new, ti, D2, G)

    return T2new


def lccsd_simple(F, I, T1old, T2old, D1, D2, ti, ng, G):
    """Time-dependent linearized coupled cluster
    singles and doubles (LCCSD) iteration.
    """
    Id = numpy.ones((ng))
    T1new = -einsum('v,ai->vai', Id, F.vo)
    T2new = -einsum('v,abij->vabij', Id, I.vvoo)

    for y in range(ng):
        cc_equations._S_S(T1new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._S_D(T1new[y], F, I, T2old[y], fac=-1.0)
        cc_equations._D_S(T2new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._D_D(T2new[y], F, I, T2old[y], fac=-1.0)

    # integrate the new T-amplitudes
    T1new = quadrature.int_tbar1(ng, T1new, ti, D1, G)
    T2new = quadrature.int_tbar2(ng, T2new, ti, D2, G)

    return T1new, T2new


def ccd_simple(F, I, T2old, D2, ti, ng, G):
    """Time-dependent coupled cluster doubles (CCD)
    iteration.
    """
    Id = numpy.ones((ng))
    T2new = -einsum('v,abij->vabij', Id, I.vvoo)

    for y in range(ng):
        cc_equations._D_D(T2new[y], F, I, T2old[y], fac=-1.0)
        cc_equations._D_DD(T2new[y], F, I, T2old[y], fac=-1.0)

    # integrate the new T-amplitudes
    T2new = quadrature.int_tbar2(ng, T2new, ti, D2, G)

    return T2new


def ccsd_simple(F, I, T1old, T2old, D1, D2, ti, ng, G):
    """Time-dependent coupled cluster singles and
    doubles (CCSD) iteration.
    """
    Id = numpy.ones((ng))
    T1new = -einsum('v,ai->vai', Id, F.vo)
    T2new = -einsum('v,abij->vabij', Id, I.vvoo)

    for y in range(ng):
        cc_equations._S_S(T1new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._S_D(T1new[y], F, I, T2old[y], fac=-1.0)
        cc_equations._S_SS(T1new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._S_SD(T1new[y], F, I, T1old[y], T2old[y], fac=-1.0)
        cc_equations._S_SSS(T1new[y], F, I, T1old[y], fac=-1.0)

        cc_equations._D_S(T2new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._D_D(T2new[y], F, I, T2old[y], fac=-1.0)
        cc_equations._D_SS(T2new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._D_SD(T2new[y], F, I, T1old[y], T2old[y], fac=-1.0)
        cc_equations._D_DD(T2new[y], F, I, T2old[y], fac=-1.0)
        cc_equations._D_SSD(T2new[y], F, I, T1old[y], T2old[y], fac=-1.0)
        cc_equations._D_SSS(T2new[y], F, I, T1old[y], fac=-1.0)
        cc_equations._D_SSSS(T2new[y], F, I, T1old[y], fac=-1.0)

    # integrate the new T-amplitudes
    T1new = quadrature.int_tbar1(ng, T1new, ti, D1, G)
    T2new = quadrature.int_tbar2(ng, T2new, ti, D2, G)

    return T1new, T2new


def ccsd_stanton(F, I, T1old, T2old, D1, D2, ti, ng, G):
    """Time-dependent coupled cluster singles and
    doubles (CCSD) iteration using Stanton-Gauss
    intermediates.
    """
    Id = numpy.ones((ng))
    T1new = -einsum('v,ai->vai', Id, F.vo)
    T2new = -einsum('v,abij->vabij', Id, I.vvoo)

    for y in range(ng):
        cc_equations._Stanton(
            T1new[y], T2new[y], F, I, T1old[y], T2old[y], fac=-1.0)

    # integrate the new T-amplitudes
    T1new = quadrature.int_tbar1(ng, T1new, ti, D1, G)
    T2new = quadrature.int_tbar2(ng, T2new, ti, D2, G)

    return T1new, T2new


def ccsd_stanton_single(ig, F, I, T1old, T2old, T1bar, T2bar, D1, D2, ti, ng, G):
    T1new = -F.vo
    T2new = -I.vvoo

    cc_equations._Stanton(T1new, T2new, F, I, T1old, T2old, fac=-1.0)

    T1bar[ig] = T1new
    T2bar[ig] = T2new

    T1new = quadrature.int_tbar1_single(ng, ig, T1bar, ti, D1, G)
    T2new = quadrature.int_tbar2_single(ng, ig, T2bar, ti, D2, G)
    return T1new, T2new


def uccsd_stanton(Fa, Fb, Ia, Ib, Iabab, T1aold, T1bold, T2aaold, T2abold,
                  T2bbold, D1a, D1b, D2aa, D2ab, D2bb, ti, ng, G):
    """Time-dependent coupled cluster singles and
    doubles (CCSD) iteration using Stanton-Gauss
    intermediates.
    """
    nva, noa = Fa.vo.shape
    nvb, nob = Fb.vo.shape

    T1a = numpy.zeros((ng, nva, noa), dtype=T1aold.dtype)
    T1b = numpy.zeros((ng, nvb, nob), dtype=T1bold.dtype)
    T2aa = numpy.zeros((ng, nva, nva, noa, noa), dtype=T2aaold.dtype)
    T2ab = numpy.zeros((ng, nva, nvb, noa, nob), dtype=T2abold.dtype)
    T2bb = numpy.zeros((ng, nvb, nvb, nob, nob), dtype=T2bbold.dtype)

    for y in range(ng):
        T1a[y] = -Fa.vo.copy()
        T1b[y] = -Fb.vo.copy()
        T2aa[y] = -Ia.vvoo.copy()
        T2bb[y] = -Ib.vvoo.copy()
        T2ab[y] = -Iabab.vvoo.copy()
        T1olds = (T1aold[y], T1bold[y])
        T2olds = (T2aaold[y], T2abold[y], T2bbold[y])
        cc_equations._u_Stanton(
            T1a[y], T1b[y], T2aa[y], T2ab[y], T2bb[y],
            Fa, Fb, Ia, Ib, Iabab, T1olds, T2olds, fac=-1.0)

    # integrate the new T-amplitudes
    T1a = quadrature.int_tbar1(ng, T1a, ti, D1a, G)
    T1b = quadrature.int_tbar1(ng, T1b, ti, D1b, G)
    T2aa = quadrature.int_tbar2(ng, T2aa, ti, D2aa, G)
    T2ab = quadrature.int_tbar2(ng, T2ab, ti, D2ab, G)
    T2bb = quadrature.int_tbar2(ng, T2bb, ti, D2bb, G)

    return (T1a, T1b), (T2aa, T2ab, T2bb)


def uccsd_stanton_single(ig, Fa, Fb, Ia, Ib, Iabab, T1a, T1b, T2aa, T2ab,
                         T2bb, T1bara, T1barb, T2baraa, T2barab, T2barbb,
                         D1a, D1b, D2aa, D2ab, D2bb, ti, ng, G):

    T1newa = -Fa.vo
    T1newb = -Fb.vo
    T2newaa = -Ia.vvoo
    T2newbb = -Ib.vvoo
    T2newab = -Iabab.vvoo

    cc_equations._u_Stanton(
        T1newa, T1newb, T2newaa, T2newab, T2newbb, Fa, Fb,
        Ia, Ib, Iabab, (T1a, T1b), (T2aa, T2ab, T2bb), fac=-1.0)

    T1bara[ig] = T1newa
    T1barb[ig] = T1newb
    T2baraa[ig] = T2newaa
    T2barab[ig] = T2newab
    T2barbb[ig] = T2newbb

    T1newa = quadrature.int_tbar1_single(ng, ig, T1bara, ti, D1a, G)
    T1newb = quadrature.int_tbar1_single(ng, ig, T1barb, ti, D1b, G)
    T2newaa = quadrature.int_tbar2_single(ng, ig, T2baraa, ti, D2aa, G)
    T2newab = quadrature.int_tbar2_single(ng, ig, T2barab, ti, D2ab, G)
    T2newbb = quadrature.int_tbar2_single(ng, ig, T2barbb, ti, D2bb, G)
    return (T1newa, T1newb), (T2newaa, T2newab, T2newbb)


def neq_ccsd_simple(Ff, Fb, F, I, T1oldf, T1oldb, T1oldi, T2oldf, T2oldb,
                    T2oldi, D1, D2, tir, tii, ngr, ngi, Gr, Gi):

    T1newf = -Ff.vo.copy()
    T1newb = -Fb.vo.copy()
    Idr = numpy.ones((ngr))
    Idi = numpy.ones((ngi))
    T1newi = -einsum('v,ai->vai', Idi, F.vo)
    T2newb = -einsum('v,abij->vabij', Idr, I.vvoo)
    T2newf = T2newb.copy()
    T2newi = -einsum('v,abij->vabij', Idi, I.vvoo)

    for y in range(ngr):
        Fftemp = one_e_blocks(Ff.oo[y], Ff.ov[y], Ff.vo[y], Ff.vv[y])
        Fbtemp = one_e_blocks(Fb.oo[y], Fb.ov[y], Fb.vo[y], Fb.vv[y])
        cc_equations._S_S(T1newf[y], Fftemp, I, T1oldf[y], fac=-1.0)
        cc_equations._S_D(T1newf[y], Fftemp, I, T2oldf[y], fac=-1.0)
        cc_equations._S_SS(T1newf[y], Fftemp, I, T1oldf[y], fac=-1.0)
        cc_equations._S_SD(T1newf[y], Fftemp, I, T1oldf[y], T2oldf[y], fac=-1.0)
        cc_equations._S_SSS(T1newf[y], Fftemp, I, T1oldf[y], fac=-1.0)

        cc_equations._D_S(T2newf[y], Fftemp, I, T1oldf[y], fac=-1.0)
        cc_equations._D_D(T2newf[y], Fftemp, I, T2oldf[y], fac=-1.0)
        cc_equations._D_SS(T2newf[y], Fftemp, I, T1oldf[y], fac=-1.0)
        cc_equations._D_SD(T2newf[y], Fftemp, I, T1oldf[y], T2oldf[y], fac=-1.0)
        cc_equations._D_DD(T2newf[y], Fftemp, I, T2oldf[y], fac=-1.0)
        cc_equations._D_SSD(T2newf[y], Fftemp, I, T1oldf[y], T2oldf[y], fac=-1.0)
        cc_equations._D_SSS(T2newf[y], Fftemp, I, T1oldf[y], fac=-1.0)
        cc_equations._D_SSSS(T2newf[y], Fftemp, I, T1oldf[y], fac=-1.0)

        cc_equations._S_S(T1newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)
        cc_equations._S_D(T1newb[y], Fbtemp, I, T2oldb[y], fac=-1.0)
        cc_equations._S_SS(T1newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)
        cc_equations._S_SD(T1newb[y], Fbtemp, I, T1oldb[y], T2oldb[y], fac=-1.0)
        cc_equations._S_SSS(T1newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)

        cc_equations._D_S(T2newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)
        cc_equations._D_D(T2newb[y], Fbtemp, I, T2oldb[y], fac=-1.0)
        cc_equations._D_SS(T2newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)
        cc_equations._D_SD(T2newb[y], Fbtemp, I, T1oldb[y], T2oldb[y], fac=-1.0)
        cc_equations._D_DD(T2newb[y], Fbtemp, I, T2oldb[y], fac=-1.0)
        cc_equations._D_SSD(T2newb[y], Fbtemp, I, T1oldb[y], T2oldb[y], fac=-1.0)
        cc_equations._D_SSS(T2newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)
        cc_equations._D_SSSS(T2newb[y], Fbtemp, I, T1oldb[y], fac=-1.0)

    for y in range(ngi):
        cc_equations._S_S(T1newi[y], F, I, T1oldi[y], fac=-1.0)
        cc_equations._S_D(T1newi[y], F, I, T2oldi[y], fac=-1.0)
        cc_equations._S_SS(T1newi[y], F, I, T1oldi[y], fac=-1.0)
        cc_equations._S_SD(T1newi[y], F, I, T1oldi[y], T2oldi[y], fac=-1.0)
        cc_equations._S_SSS(T1newi[y], F, I, T1oldi[y], fac=-1.0)

        cc_equations._D_S(T2newi[y], F, I, T1oldi[y], fac=-1.0)
        cc_equations._D_D(T2newi[y], F, I, T2oldi[y], fac=-1.0)
        cc_equations._D_SS(T2newi[y], F, I, T1oldi[y], fac=-1.0)
        cc_equations._D_SD(T2newi[y], F, I, T1oldi[y], T2oldi[y], fac=-1.0)
        cc_equations._D_DD(T2newi[y], F, I, T2oldi[y], fac=-1.0)
        cc_equations._D_SSD(T2newi[y], F, I, T1oldi[y], T2oldi[y], fac=-1.0)
        cc_equations._D_SSS(T2newi[y], F, I, T1oldi[y], fac=-1.0)
        cc_equations._D_SSSS(T2newi[y], F, I, T1oldi[y], fac=-1.0)

    T1newf, T1newb, T1newi = quadrature.int_tbar1_keldysh(
        ngr, ngi, T1newf, T1newb, T1newi, tir, tii, D1, Gr, Gi)
    T2newf, T2newb, T2newi = quadrature.int_tbar2_keldysh(
        ngr, ngi, T2newf, T2newb, T2newi, tir, tii, D2, Gr, Gi)

    return T1newf, T1newb, T1newi, T2newf, T2newb, T2newi


def neq_ccsd_stanton(Ff, Fb, F, I, T1oldf, T1oldb, T1oldi, T2oldf, T2oldb,
                     T2oldi, D1, D2, tir, tii, ngr, ngi, Gr, Gi):

    T1newf = -Ff.vo.copy()
    T1newb = -Fb.vo.copy()
    Idr = numpy.ones((ngr))
    Idi = numpy.ones((ngi))
    T1newi = -einsum('v,ai->vai', Idi, F.vo)
    T2newb = -einsum('v,abij->vabij', Idr, I.vvoo)
    T2newf = T2newb.copy()
    T2newi = -einsum('v,abij->vabij', Idi, I.vvoo)

    for y in range(ngr):
        Fftemp = one_e_blocks(Ff.oo[y], Ff.ov[y], Ff.vo[y], Ff.vv[y])
        Fbtemp = one_e_blocks(Fb.oo[y], Fb.ov[y], Fb.vo[y], Fb.vv[y])
        cc_equations._Stanton(T1newf[y], T2newf[y], Fftemp, I, T1oldf[y], T2oldf[y], fac=-1.0)
        cc_equations._Stanton(T1newb[y], T2newb[y], Fbtemp, I, T1oldb[y], T2oldb[y], fac=-1.0)
    for y in range(ngi):
        cc_equations._Stanton(T1newi[y], T2newi[y], F, I, T1oldi[y], T2oldi[y], fac=-1.0)

    T1newf, T1newb, T1newi = quadrature.int_tbar1_keldysh(
        ngr, ngi, T1newf, T1newb, T1newi, tir, tii, D1, Gr, Gi)
    T2newf, T2newb, T2newi = quadrature.int_tbar2_keldysh(
        ngr, ngi, T2newf, T2newb, T2newi, tir, tii, D2, Gr, Gi)

    return T1newf, T1newb, T1newi, T2newf, T2newb, T2newi


def lccd_lambda_simple(F, I, T2old, L2old, D2, ti, ng, g, G, beta):
    """Time-dependent linearized coupled cluster doubles (LCCD)
    Lambda iteration.
    """
    # integrate old lambda amplitudes
    L2int = quadrature.int_L2(ng, L2old, ti, D2, g, G)

    # initialize lambda amplitudes
    L2 = numpy.zeros(L2old.shape, dtype=L2old.dtype)

    # amplitude term
    for y in range(ng):
        cc_equations._LD_LD(L2[y], F, I, L2int[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L2 -= (1.0/beta)*einsum('v,ijab->vijab', Id, I.oovv)

    return L2


def lccsd_lambda_simple(F, I, T1old, T2old, L1old, L2old, D1, D2, ti, ng, g, G, beta):
    """Time-dependent linearized coupled cluster singles and doubles (LCCSD)
    Lambda iteration.
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng, L1old, ti, D1, g, G)
    L2int = quadrature.int_L2(ng, L2old, ti, D2, g, G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape, dtype=L1old.dtype)
    L2 = numpy.zeros(L2old.shape, dtype=L2old.dtype)

    # amplitude term
    for y in range(ng):
        cc_equations._LS_LS(L1[y], F, I, L1int[y], fac=-1.0)
        cc_equations._LS_LD(L1[y], F, I, L2int[y], fac=-1.0)

        cc_equations._LD_LS(L2[y], F, I, L1int[y], fac=-1.0)
        cc_equations._LD_LD(L2[y], F, I, L2int[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 -= einsum('v,ia->via', Id, F.ov)
    L2 -= einsum('v,ijab->vijab', Id, I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y], I, T1old[y], fac=-1.0)

    return L1, L2


def ccsd_lambda_simple(F, I, T1old, T2old, L1old, L2old, D1, D2, ti, ng, g, G, beta):
    """Time-dependent coupled cluster singles and doubles (CCSD)
    Lambda iteration.
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng, L1old, ti, D1, g, G)
    L2int = quadrature.int_L2(ng, L2old, ti, D2, g, G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape, dtype=L1old.dtype)
    L2 = numpy.zeros(L2old.shape, dtype=L2old.dtype)

    # amplitude term
    for y in range(ng):
        cc_equations._LS_LS(L1[y], F, I, L1int[y], fac=-1.0)
        cc_equations._LS_LSTS(L1[y], F, I, L1int[y], T1old[y], fac=-1.0)
        cc_equations._LS_LSTD(L1[y], I, L1int[y], T2old[y], fac=-1.0)
        cc_equations._LS_LSTSS(L1[y], I, L1int[y], T1old[y], fac=-1.0)
        cc_equations._LS_LD(L1[y], F, I, L2int[y], fac=-1.0)
        cc_equations._LS_LDTS(L1[y], F, I, L2int[y], T1old[y], fac=-1.0)
        cc_equations._LS_LDTD(L1[y], F, I, L2int[y], T2old[y], fac=-1.0)
        cc_equations._LS_LDTSS(L1[y], F, I, L2int[y], T1old[y], fac=-1.0)
        cc_equations._LS_LDTSD(L1[y], I, L2int[y], T1old[y], T2old[y], fac=-1.0)
        cc_equations._LS_LDTSSS(L1[y], I, L2int[y], T1old[y], fac=-1.0)

        cc_equations._LD_LS(L2[y], F, I, L1int[y], fac=-1.0)
        cc_equations._LD_LSTS(L2[y], F, I, L1int[y], T1old[y], fac=-1.0)
        cc_equations._LD_LD(L2[y], F, I, L2int[y], fac=-1.0)
        cc_equations._LD_LDTS(L2[y], F, I, L2int[y], T1old[y], fac=-1.0)
        cc_equations._LD_LDTD(L2[y], I, L2int[y], T2old[y], fac=-1.0)
        cc_equations._LD_LDTSS(L2[y], F, I, L2int[y], T1old[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 -= einsum('v,ia->via', Id, F.ov)
    L2 -= einsum('v,ijab->vijab', Id, I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y], I, T1old[y], fac=-1.0)

    return L1, L2


def ccsd_lambda_opt(F, I, T1old, T2old, L1old, L2old, D1, D2, ti, ng, g, G, beta):
    """Time-dependent coupled cluster singles and doubles (CCSD)
    Lambda iteration with intermediates.
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng, L1old, ti, D1, g, G)
    L2int = quadrature.int_L2(ng, L2old, ti, D2, g, G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape, dtype=L1old.dtype)
    L2 = numpy.zeros(L2old.shape, dtype=L2old.dtype)

    # amplitude term
    for y in range(ng):
        cc_equations._Lambda_opt(L1[y], L2[y], F, I, L1int[y], L2int[y],
                                 T1old[y], T2old[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 -= einsum('v,ia->via', Id, F.ov)
    L2 -= einsum('v,ijab->vijab', Id, I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y], I, T1old[y], fac=-1.0)

    return L1, L2


def uccsd_lambda_opt(Fa, Fb, Ia, Ib, Iabab, T1aold, T1bold, T2aaold, T2abold,
                     T2bbold, L1aold, L1bold, L2aaold, L2abold, L2bbold, D1a,
                     D1b, D2aa, D2ab, D2bb, ti, ng, g, G, beta):
    """Time-dependent coupled cluster singles and doubles (CCSD)
    Lambda iteration with intermediates.
    """
    nt, nva, noa = T1aold.shape
    nt, nvb, nob = T1bold.shape
    assert(nt == ng)

    # integrate old lambda amplitudes
    L1aint = quadrature.int_L1(ng, L1aold, ti, D1a, g, G)
    L1bint = quadrature.int_L1(ng, L1bold, ti, D1b, g, G)
    L2aaint = quadrature.int_L2(ng, L2aaold, ti, D2aa, g, G)
    L2abint = quadrature.int_L2(ng, L2abold, ti, D2ab, g, G)
    L2bbint = quadrature.int_L2(ng, L2bbold, ti, D2bb, g, G)

    # initialize lambda amplitudes
    L1a = numpy.zeros((ng, noa, nva), dtype=L1aold.dtype)
    L1b = numpy.zeros((ng, nob, nvb), dtype=L1bold.dtype)
    L2aa = numpy.zeros((ng, noa, noa, nva, nva), dtype=L2aaold.dtype)
    L2ab = numpy.zeros((ng, noa, nob, nva, nvb), dtype=L2abold.dtype)
    L2bb = numpy.zeros((ng, nob, nob, nvb, nvb), dtype=L2bbold.dtype)

    # amplitude term
    for y in range(ng):
        L1olds = (L1aint[y], L1bint[y])
        T1olds = (T1aold[y], T1bold[y])
        L2olds = (L2aaint[y], L2abint[y], L2bbint[y])
        T2olds = (T2aaold[y], T2abold[y], T2bbold[y])
        cc_equations._uccsd_Lambda_opt(
            L1a[y], L1b[y], L2aa[y], L2ab[y], L2bb[y],
            Fa, Fb, Ia, Ib, Iabab, L1olds, L2olds,
            T1olds, T2olds, fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1a -= einsum('v,ia->via', Id, Fa.ov)
    L1b -= einsum('v,ia->via', Id, Fb.ov)
    L2aa -= einsum('v,ijab->vijab', Id, Ia.oovv)
    L2ab -= einsum('v,ijab->vijab', Id, Iabab.oovv)
    L2bb -= einsum('v,ijab->vijab', Id, Ib.oovv)
    for y in range(ng):
        T1olds = (T1aold[y], T1bold[y])
        cc_equations._u_LS_TS(L1a[y], L1b[y], Ia, Ib, Iabab, T1olds[0], T1olds[1], fac=-1.0)

    return L1a, L1b, L2aa, L2ab, L2bb


def ft_ccsd_lambda_int(F, I, T1old, T2old):
    ng = T1old.shape[0]
    intor = []
    for y in range(ng):
        intor.append(cc_equations.lambda_int(F, I, T1old[y], T2old[y]))
    return intor


def ccsd_lambda_opt_int(F, I, T1old, T2old, L1old, L2old, intor, D1, D2, ti, ng, g, G, beta):
    """Time-dependent coupled cluster singles and doubles (CCSD)
    Lambda iteration with precomputed intermediates.
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng, L1old, ti, D1, g, G)
    L2int = quadrature.int_L2(ng, L2old, ti, D2, g, G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape, dtype=L1old.dtype)
    L2 = numpy.zeros(L2old.shape, dtype=L2old.dtype)

    # amplitude term
    for y in range(ng):
        cc_equations._Lambda_opt_int(L1[y], L2[y], F, I, L1int[y], L2int[y],
                                     T1old[y], T2old[y], intor[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 -= einsum('v,ia->via', Id, F.ov)
    L2 -= einsum('v,ijab->vijab', Id, I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y], I, T1old[y], fac=-1.0)

    return L1, L2


def ccd_lambda_guess(I, beta, ng):
    Id = numpy.ones((ng))
    L2 = (1.0/beta)*einsum('v,ijab->vijab', Id, I.oovv)
    return L2


def ccsd_lambda_guess(F, I, T1old, beta, ng):
    """Time-dependent coupled cluster singles and doubles (CCSD)
    Lambda guess.
    """
    Id = numpy.ones((ng))
    L1 = (1.0/beta)*einsum('v,ia->via', Id, F.ov)
    L2 = (1.0/beta)*einsum('v,ijab->vijab', Id, I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y], I, T1old[y], fac=(1.0/beta))

    return L1, L2


def uccsd_lambda_guess(Fa, Fb, Ia, Ib, Iabab, T1aold, T1bold, beta, ng):
    Id = numpy.ones((ng))
    L1a = (1.0/beta)*einsum('v,ia->via', Id, Fa.ov)
    L1b = (1.0/beta)*einsum('v,ia->via', Id, Fb.ov)
    L2aa = (1.0/beta)*einsum('v,ijab->vijab', Id, Ia.oovv)
    L2ab = (1.0/beta)*einsum('v,ijab->vijab', Id, Iabab.oovv)
    L2bb = (1.0/beta)*einsum('v,ijab->vijab', Id, Ib.oovv)
    for y in range(ng):
        T1olds = (T1aold[y], T1bold[y])
        cc_equations._u_LS_TS(L1a[y], L1b[y], Ia, Ib, Iabab, T1olds[0], T1olds[1])

    return L1a, L1b, L2aa, L2ab, L2bb


def uccd_lambda_guess(Ia, Ib, Iabab, beta, ng):
    Id = numpy.ones((ng))
    L2aa = (1.0/beta)*einsum('v,ijab->vijab', Id, Ia.oovv)
    L2ab = (1.0/beta)*einsum('v,ijab->vijab', Id, Iabab.oovv)
    L2bb = (1.0/beta)*einsum('v,ijab->vijab', Id, Ib.oovv)

    return L2aa, L2ab, L2bb


def neq_lambda_simple(Ff, Fb, F, I, L1oldf, L1oldb, L1oldi, L2oldf, L2oldb,
                      L2oldi, T1oldf, T1oldb, T1oldi, T2oldf, T2oldb, T2oldi,
                      D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi):

    # integrate old lambda amplitudes
    L1intf, L1intb, L1inti = quadrature.int_L1_keldysh(
        ngr, ngi, L1oldf, L1oldb, L1oldi, tir, tii, D1, gr, gi, Gr, Gi)
    L2intf, L2intb, L2inti = quadrature.int_L2_keldysh(
        ngr, ngi, L2oldf, L2oldb, L2oldi, tir, tii, D2, gr, gi, Gr, Gi)

    # initialize lambda amplitudes
    L1f = numpy.zeros(L1oldf.shape, dtype=complex)
    L1b = numpy.zeros(L1oldb.shape, dtype=complex)
    L1i = numpy.zeros(L1oldi.shape, dtype=complex)
    L2f = numpy.zeros(L2oldf.shape, dtype=complex)
    L2b = numpy.zeros(L2oldb.shape, dtype=complex)
    L2i = numpy.zeros(L2oldi.shape, dtype=complex)

    # amplitude term
    for y in range(ngi):
        cc_equations._LS_LS(L1i[y], F, I, L1inti[y], fac=-1.0)
        cc_equations._LS_LSTS(L1i[y], F, I, L1inti[y], T1oldi[y], fac=-1.0)
        cc_equations._LS_LSTD(L1i[y], I, L1inti[y], T2oldi[y], fac=-1.0)
        cc_equations._LS_LSTSS(L1i[y], I, L1inti[y], T1oldi[y], fac=-1.0)
        cc_equations._LS_LD(L1i[y], F, I, L2inti[y], fac=-1.0)
        cc_equations._LS_LDTS(L1i[y], F, I, L2inti[y], T1oldi[y], fac=-1.0)
        cc_equations._LS_LDTD(L1i[y], F, I, L2inti[y], T2oldi[y], fac=-1.0)
        cc_equations._LS_LDTSS(L1i[y], F, I, L2inti[y], T1oldi[y], fac=-1.0)
        cc_equations._LS_LDTSD(L1i[y], I, L2inti[y], T1oldi[y], T2oldi[y], fac=-1.0)
        cc_equations._LS_LDTSSS(L1i[y], I, L2inti[y], T1oldi[y], fac=-1.0)

        cc_equations._LD_LS(L2i[y], F, I, L1inti[y], fac=-1.0)
        cc_equations._LD_LSTS(L2i[y], F, I, L1inti[y], T1oldi[y], fac=-1.0)
        cc_equations._LD_LD(L2i[y], F, I, L2inti[y], fac=-1.0)
        cc_equations._LD_LDTS(L2i[y], F, I, L2inti[y], T1oldi[y], fac=-1.0)
        cc_equations._LD_LDTD(L2i[y], I, L2inti[y], T2oldi[y], fac=-1.0)
        cc_equations._LD_LDTSS(L2i[y], F, I, L2inti[y], T1oldi[y], fac=-1.0)

    for y in range(ngr):
        Fbtemp = one_e_blocks(Fb.oo[y], Fb.ov[y], Fb.vo[y], Fb.vv[y])
        cc_equations._LS_LS(L1b[y], Fbtemp, I, L1intb[y], fac=-1.0)
        cc_equations._LS_LSTS(L1b[y], Fbtemp, I, L1intb[y], T1oldb[y], fac=-1.0)
        cc_equations._LS_LSTD(L1b[y], I, L1intb[y], T2oldb[y], fac=-1.0)
        cc_equations._LS_LSTSS(L1b[y], I, L1intb[y], T1oldb[y], fac=-1.0)
        cc_equations._LS_LD(L1b[y], Fbtemp, I, L2intb[y], fac=-1.0)
        cc_equations._LS_LDTS(L1b[y], Fbtemp, I, L2intb[y], T1oldb[y], fac=-1.0)
        cc_equations._LS_LDTD(L1b[y], Fbtemp, I, L2intb[y], T2oldb[y], fac=-1.0)
        cc_equations._LS_LDTSS(L1b[y], Fbtemp, I, L2intb[y], T1oldb[y], fac=-1.0)
        cc_equations._LS_LDTSD(L1b[y], I, L2intb[y], T1oldb[y], T2oldb[y], fac=-1.0)
        cc_equations._LS_LDTSSS(L1b[y], I, L2intb[y], T1oldb[y], fac=-1.0)

        cc_equations._LD_LS(L2b[y], Fbtemp, I, L1intb[y], fac=-1.0)
        cc_equations._LD_LSTS(L2b[y], Fbtemp, I, L1intb[y], T1oldb[y], fac=-1.0)
        cc_equations._LD_LD(L2b[y], Fbtemp, I, L2intb[y], fac=-1.0)
        cc_equations._LD_LDTS(L2b[y], Fbtemp, I, L2intb[y], T1oldb[y], fac=-1.0)
        cc_equations._LD_LDTD(L2b[y], I, L2intb[y], T2oldb[y], fac=-1.0)
        cc_equations._LD_LDTSS(L2b[y], Fbtemp, I, L2intb[y], T1oldb[y], fac=-1.0)

        Fftemp = one_e_blocks(Ff.oo[y], Ff.ov[y], Ff.vo[y], Ff.vv[y])
        cc_equations._LS_LS(L1f[y], Fftemp, I, L1intf[y], fac=-1.0)
        cc_equations._LS_LSTS(L1f[y], Fftemp, I, L1intf[y], T1oldf[y], fac=-1.0)
        cc_equations._LS_LSTD(L1f[y], I, L1intf[y], T2oldf[y], fac=-1.0)
        cc_equations._LS_LSTSS(L1f[y], I, L1intf[y], T1oldf[y], fac=-1.0)
        cc_equations._LS_LD(L1f[y], Fftemp, I, L2intf[y], fac=-1.0)
        cc_equations._LS_LDTS(L1f[y], Fftemp, I, L2intf[y], T1oldf[y], fac=-1.0)
        cc_equations._LS_LDTD(L1f[y], Fftemp, I, L2intf[y], T2oldf[y], fac=-1.0)
        cc_equations._LS_LDTSS(L1f[y], Fftemp, I, L2intf[y], T1oldf[y], fac=-1.0)
        cc_equations._LS_LDTSD(L1f[y], I, L2intf[y], T1oldf[y], T2oldf[y], fac=-1.0)
        cc_equations._LS_LDTSSS(L1f[y], I, L2intf[y], T1oldf[y], fac=-1.0)

        cc_equations._LD_LS(L2f[y], Fftemp, I, L1intf[y], fac=-1.0)
        cc_equations._LD_LSTS(L2f[y], Fftemp, I, L1intf[y], T1oldf[y], fac=-1.0)
        cc_equations._LD_LD(L2f[y], Fftemp, I, L2intf[y], fac=-1.0)
        cc_equations._LD_LDTS(L2f[y], Fftemp, I, L2intf[y], T1oldf[y], fac=-1.0)
        cc_equations._LD_LDTD(L2f[y], I, L2intf[y], T2oldf[y], fac=-1.0)
        cc_equations._LD_LDTSS(L2f[y], Fftemp, I, L2intf[y], T1oldf[y], fac=-1.0)

    # energy term
    for y in range(ngr):
        L1f[y] += Ff.ov[y]
        L1b[y] += Fb.ov[y]
        L2f[y] += I.oovv
        L2b[y] += I.oovv

        cc_equations._LS_TS(L1f[y], I, T1oldf[y], fac=1.0)
        cc_equations._LS_TS(L1b[y], I, T1oldb[y], fac=1.0)

    for y in range(ngi):
        L1i[y] += F.ov
        L2i[y] += I.oovv
        cc_equations._LS_TS(L1i[y], I, T1oldi[y], fac=1.0)

    return L1f, L1b, L1i, L2f, L2b, L2i


def neq_lambda_opt(Ff, Fb, F, I, L1oldf, L1oldb, L1oldi, L2oldf, L2oldb,
                   L2oldi, T1oldf, T1oldb, T1oldi, T2oldf, T2oldb, T2oldi,
                   D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi):

    # integrate old lambda amplitudes
    L1intf, L1intb, L1inti = quadrature.int_L1_keldysh(
        ngr, ngi, L1oldf, L1oldb, L1oldi, tir, tii, D1, gr, gi, Gr, Gi)
    L2intf, L2intb, L2inti = quadrature.int_L2_keldysh(
        ngr, ngi, L2oldf, L2oldb, L2oldi, tir, tii, D2, gr, gi, Gr, Gi)

    # initialize lambda amplitudes
    L1f = numpy.zeros(L1oldf.shape, dtype=complex)
    L1b = numpy.zeros(L1oldb.shape, dtype=complex)
    L1i = numpy.zeros(L1oldi.shape, dtype=complex)
    L2f = numpy.zeros(L2oldf.shape, dtype=complex)
    L2b = numpy.zeros(L2oldb.shape, dtype=complex)
    L2i = numpy.zeros(L2oldi.shape, dtype=complex)

    # amplitude term
    for y in range(ngr):
        Fftemp = one_e_blocks(Ff.oo[y], Ff.ov[y], Ff.vo[y], Ff.vv[y])
        cc_equations._Lambda_opt(L1f[y], L2f[y], Fftemp, I, L1intf[y],
                                 L2intf[y], T1oldf[y], T2oldf[y], fac=-1.0)
        Fbtemp = one_e_blocks(Fb.oo[y], Fb.ov[y], Fb.vo[y], Fb.vv[y])
        cc_equations._Lambda_opt(L1b[y], L2b[y], Fbtemp, I, L1intb[y],
                                 L2intb[y], T1oldb[y], T2oldb[y], fac=-1.0)

    for y in range(ngi):
        cc_equations._Lambda_opt(L1i[y], L2i[y], F, I, L1inti[y], L2inti[y],
                                 T1oldi[y], T2oldi[y], fac=-1.0)

    # energy term
    for y in range(ngr):
        L1f[y] += Ff.ov[y]
        L1b[y] += Fb.ov[y]
        L2f[y] += I.oovv
        L2b[y] += I.oovv

        cc_equations._LS_TS(L1f[y], I, T1oldf[y], fac=1.0)
        cc_equations._LS_TS(L1b[y], I, T1oldb[y], fac=1.0)

    for y in range(ngi):
        L1i[y] += F.ov
        L2i[y] += I.oovv
        cc_equations._LS_TS(L1i[y], I, T1oldi[y], fac=1.0)

    return L1f, L1b, L1i, L2f, L2b, L2i


def ccd_lambda_simple(F, I, T2old, L2old, D2, ti, ng, g, G, beta):
    """Time-dependent coupled cluster doubles (CCD)
    Lambda iteration.
    """
    # integrate old lambda amplitudes
    L2int = quadrature.int_L2(ng, L2old, ti, D2, g, G)

    # initialize lambda amplitudes
    L2 = numpy.zeros(L2old.shape, dtype=L2old.dtype)

    # amplitude term
    for y in range(ng):
        cc_equations._LD_LD(L2[y], F, I, L2int[y], fac=-1.0)
        cc_equations._LD_LDTD(L2[y], I, L2int[y], T2old[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L2 -= einsum('v,ijab->vijab', Id, I.oovv)

    return L2


def ccsd_1rdm(T1, T2, L1, L2, D1, D2, ti, ng, g, G):

    # integrate the new L-amplitudes
    L1new = quadrature.int_L1(ng, L1, ti, D1, g, G)
    L2new = quadrature.int_L2(ng, L2, ti, D2, g, G)
    nt, nv, no = T1.shape
    assert(nt == ng)

    # compute response densities
    pia = numpy.einsum('sia,s->ia', L1new, g)
    pba = numpy.zeros((nv, nv), dtype=T1.dtype)
    pji = numpy.zeros((no, no), dtype=T1.dtype)
    pai = numpy.zeros((nv, no), dtype=T1.dtype)
    for i in range(nt):
        pba += g[i]*cc_equations.ccsd_1rdm_ba_opt(T1[i], T2[i], L1new[i], L2new[i])
        pji += g[i]*cc_equations.ccsd_1rdm_ji_opt(T1[i], T2[i], L1new[i], L2new[i])
        pai += g[i]*cc_equations.ccsd_1rdm_ai_opt(T1[i], T2[i], L1new[i], L2new[i])

    return pia, pba, pji, pai


def ccsd_2rdm(T1, T2, L1, L2, D1, D2, ti, ng, g, G):
    nt, nv, no = T1.shape
    assert(nt == ng)

    # integrate the new L-amplitudes
    L1new = quadrature.int_L1(ng, L1, ti, D1, g, G)
    L2new = quadrature.int_L2(ng, L2, ti, D2, g, G)

    # compute response densities
    Pcdab = numpy.zeros((nv, nv, nv, nv), dtype=T2.dtype)
    Pciab = numpy.zeros((nv, no, nv, nv), dtype=T2.dtype)
    Pbcai = numpy.zeros((nv, nv, nv, no), dtype=T2.dtype)
    Pijab = einsum('sijab,s->ijab', L2new, g)
    Pbjai = numpy.zeros((nv, no, nv, no), dtype=T2.dtype)
    Pabij = numpy.zeros((nv, nv, no, no), dtype=T2.dtype)
    Pjkai = numpy.zeros((no, no, nv, no), dtype=T2.dtype)
    Pkaij = numpy.zeros((no, nv, no, no), dtype=T2.dtype)
    Pklij = numpy.zeros((no, no, no, no), dtype=T2.dtype)
    for i in range(nt):
        Pcdab += g[i]*cc_equations.ccsd_2rdm_cdab_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pciab += g[i]*cc_equations.ccsd_2rdm_ciab_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pbcai += g[i]*cc_equations.ccsd_2rdm_bcai_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pbjai += g[i]*cc_equations.ccsd_2rdm_bjai_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pabij += g[i]*cc_equations.ccsd_2rdm_abij_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pjkai += g[i]*cc_equations.ccsd_2rdm_jkai_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pkaij += g[i]*cc_equations.ccsd_2rdm_kaij_opt(T1[i], T2[i], L1new[i], L2new[i])
        Pklij += g[i]*cc_equations.ccsd_2rdm_klij_opt(T1[i], T2[i], L1new[i], L2new[i])

    return (Pcdab, Pciab, Pbcai, Pijab, Pbjai, Pabij, Pjkai, Pkaij, Pklij)


def uccsd_1rdm(T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb,
               D1a, D1b, D2aa, D2ab, D2bb, ti, ng, g, G):

    # integrate the new L-amplitudes
    nt1, nva, noa = T1a.shape
    nt2, nvb, nob = T1b.shape
    assert(nt1 == ng and nt2 == ng)

    L1anew = quadrature.int_L1(ng, L1a, ti, D1a, g, G)
    L1bnew = quadrature.int_L1(ng, L1b, ti, D1b, g, G)
    L2aanew = quadrature.int_L2(ng, L2aa, ti, D2aa, g, G)
    L2abnew = quadrature.int_L2(ng, L2ab, ti, D2ab, g, G)
    L2bbnew = quadrature.int_L2(ng, L2bb, ti, D2bb, g, G)

    # spin blocks of the response densities
    pia = numpy.einsum('sia,s->ia', L1anew, g)
    pIA = numpy.einsum('sia,s->ia', L1bnew, g)
    pba = numpy.zeros((nva, nva), dtype=T1a.dtype)
    pBA = numpy.zeros((nvb, nvb), dtype=T1b.dtype)
    pji = numpy.zeros((noa, noa), dtype=T1a.dtype)
    pJI = numpy.zeros((nob, nob), dtype=T1b.dtype)
    pai = numpy.zeros((nva, noa), dtype=T1a.dtype)
    pAI = numpy.zeros((nvb, nob), dtype=T1b.dtype)
    for i in range(ng):
        pba_tot = cc_equations.uccsd_1rdm_ba(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        pba += g[i]*pba_tot[0]
        pBA += g[i]*pba_tot[1]

        pji_tot = cc_equations.uccsd_1rdm_ji(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        pji += g[i]*pji_tot[0]
        pJI += g[i]*pji_tot[1]

        pai_tot = cc_equations.uccsd_1rdm_ai(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        pai += g[i]*pai_tot[0]
        pAI += g[i]*pai_tot[1]

    return (pia, pIA), (pba, pBA), (pji, pJI), (pai, pAI)


def uccsd_2rdm(T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb,
               D1a, D1b, D2aa, D2ab, D2bb, ti, ng, g, G):
    nt1, nva, noa = T1a.shape
    nt2, nvb, nob = T1b.shape
    assert(nt1 == ng and nt2 == ng)

    # integrate the new L-amplitudes
    L1anew = quadrature.int_L1(ng, L1a, ti, D1a, g, G)
    L1bnew = quadrature.int_L1(ng, L1b, ti, D1b, g, G)
    L2aanew = quadrature.int_L2(ng, L2aa, ti, D2aa, g, G)
    L2abnew = quadrature.int_L2(ng, L2ab, ti, D2ab, g, G)
    L2bbnew = quadrature.int_L2(ng, L2bb, ti, D2bb, g, G)

    # spin blocks of the response densities
    Pcdab = numpy.zeros((nva, nva, nva, nva), dtype=T2aa.dtype)
    Pciab = numpy.zeros((nva, noa, nva, nva), dtype=T2aa.dtype)
    Pbcai = numpy.zeros((nva, nva, nva, noa), dtype=T2aa.dtype)
    Pijab = einsum('sijab,s->ijab', L2aanew, g)
    Pbjai = numpy.zeros((nva, noa, nva, noa), dtype=T2aa.dtype)
    Pabij = numpy.zeros((nva, nva, noa, noa), dtype=T2aa.dtype)
    Pjkai = numpy.zeros((noa, noa, nva, noa), dtype=T2aa.dtype)
    Pkaij = numpy.zeros((noa, nva, noa, noa), dtype=T2aa.dtype)
    Pklij = numpy.zeros((noa, noa, noa, noa), dtype=T2aa.dtype)

    PCDAB = numpy.zeros((nvb, nvb, nvb, nvb), dtype=T2bb.dtype)
    PCIAB = numpy.zeros((nvb, nob, nvb, nvb), dtype=T2bb.dtype)
    PBCAI = numpy.zeros((nvb, nvb, nvb, nob), dtype=T2bb.dtype)
    PIJAB = einsum('sijab,s->ijab', L2bbnew, g)
    PBJAI = numpy.zeros((nvb, nob, nvb, nob), dtype=T2bb.dtype)
    PABIJ = numpy.zeros((nvb, nvb, nob, nob), dtype=T2bb.dtype)
    PJKAI = numpy.zeros((nob, nob, nvb, nob), dtype=T2bb.dtype)
    PKAIJ = numpy.zeros((nob, nvb, nob, nob), dtype=T2bb.dtype)
    PKLIJ = numpy.zeros((nob, nob, nob, nob), dtype=T2bb.dtype)

    PcDaB = numpy.zeros((nva, nvb, nva, nvb), dtype=T2ab.dtype)
    PcIaB = numpy.zeros((nva, nob, nva, nvb), dtype=T2ab.dtype)
    PbCaI = numpy.zeros((nva, nvb, nva, nob), dtype=T2ab.dtype)
    PiJaB = einsum('sijab,s->ijab', L2abnew, g)
    PbJaI = numpy.zeros((nva, nob, nva, nob), dtype=T2ab.dtype)
    PaBiJ = numpy.zeros((nva, nvb, noa, nob), dtype=T2ab.dtype)
    PjKaI = numpy.zeros((noa, nob, nva, nob), dtype=T2ab.dtype)
    PkAiJ = numpy.zeros((noa, nvb, noa, nob), dtype=T2ab.dtype)
    PkLiJ = numpy.zeros((noa, nob, noa, nob), dtype=T2ab.dtype)

    PCiAb = numpy.zeros((nvb, noa, nvb, nva), dtype=T2ab.dtype)
    PBcAi = numpy.zeros((nvb, nva, nvb, noa), dtype=T2ab.dtype)
    PJkAi = numpy.zeros((nob, noa, nvb, noa), dtype=T2ab.dtype)
    PKaIj = numpy.zeros((nob, nva, nob, noa), dtype=T2ab.dtype)

    PbJAi = numpy.zeros((nva, nob, nvb, noa), dtype=T2ab.dtype)
    PBjaI = numpy.zeros((nvb, noa, nva, nob), dtype=T2ab.dtype)
    PBjAi = numpy.zeros((nvb, noa, nvb, noa), dtype=T2ab.dtype)

    # compute response densities
    for i in range(ng):
        Pcdab_tot = cc_equations.uccsd_2rdm_cdab(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pcdab += g[i]*Pcdab_tot[0]
        PCDAB += g[i]*Pcdab_tot[1]
        PcDaB += g[i]*Pcdab_tot[2]

        Pciab_tot = cc_equations.uccsd_2rdm_ciab(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pciab += g[i]*Pciab_tot[0]
        PCIAB += g[i]*Pciab_tot[1]
        PcIaB += g[i]*Pciab_tot[2]
        PCiAb += g[i]*Pciab_tot[3]

        Pbcai_tot = cc_equations.uccsd_2rdm_bcai(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pbcai += g[i]*Pbcai_tot[0]
        PBCAI += g[i]*Pbcai_tot[1]
        PbCaI += g[i]*Pbcai_tot[2]
        PBcAi += g[i]*Pbcai_tot[3]

        Pbjai_tot = cc_equations.uccsd_2rdm_bjai(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pbjai += g[i]*Pbjai_tot[0]
        PBJAI += g[i]*Pbjai_tot[1]
        PbJaI += g[i]*Pbjai_tot[2]
        PbJAi += g[i]*Pbjai_tot[3]
        PBjaI += g[i]*Pbjai_tot[4]
        PBjAi += g[i]*Pbjai_tot[5]

        Pabij_tot = cc_equations.uccsd_2rdm_abij(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pabij += g[i]*Pabij_tot[0]
        PABIJ += g[i]*Pabij_tot[1]
        PaBiJ += g[i]*Pabij_tot[2]

        Pjkai_tot = cc_equations.uccsd_2rdm_jkai(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pjkai += g[i]*Pjkai_tot[0]
        PJKAI += g[i]*Pjkai_tot[1]
        PjKaI += g[i]*Pjkai_tot[2]
        PJkAi += g[i]*Pjkai_tot[3]

        Pkaij_tot = cc_equations.uccsd_2rdm_kaij(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pkaij += g[i]*Pkaij_tot[0]
        PKAIJ += g[i]*Pkaij_tot[1]
        PkAiJ += g[i]*Pkaij_tot[2]
        PKaIj += g[i]*Pkaij_tot[3]

        Pklij_tot = cc_equations.uccsd_2rdm_klij(
            T1a[i], T1b[i], T2aa[i], T2ab[i], T2bb[i],
            L1anew[i], L1bnew[i], L2aanew[i], L2abnew[i], L2bbnew[i])
        Pklij += g[i]*Pklij_tot[0]
        PKLIJ += g[i]*Pklij_tot[1]
        PkLiJ += g[i]*Pklij_tot[2]

    return ((Pcdab, PCDAB, PcDaB),
            (Pciab, PCIAB, PcIaB, PCiAb),
            (Pbcai, PBCAI, PbCaI, PBcAi),
            (Pijab, PIJAB, PiJaB),
            (Pbjai, PBJAI, PbJaI, PbJAi, PBjaI, PBjAi),
            (Pabij, PABIJ, PaBiJ),
            (Pjkai, PJKAI, PjKaI, PJkAi),
            (Pkaij, PKAIJ, PkAiJ, PKaIj),
            (Pklij, PKLIJ, PkLiJ))


def neq_1rdm(T1f, T1b, T1i, T2f, T2b, T2i, L1f, L1b, L1i, L2f, L2b, L2i,
             D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi):

    # integrate the new L-amplitudes
    L1intf, L1intb, L1inti = quadrature.int_L1_keldysh(
        ngr, ngi, L1f, L1b, L1i, tir, tii, D1, gr, gi, Gr, Gi)
    L2intf, L2intb, L2inti = quadrature.int_L2_keldysh(
        ngr, ngi, L2f, L2b, L2i, tir, tii, D2, gr, gi, Gr, Gi)
    ntr, nv, no = T1f.shape
    assert(ntr == ngr)

    # compute response densities
    piaf = -L1intf
    pbaf = numpy.zeros((ntr, nv, nv), dtype=complex)
    pjif = numpy.zeros((ntr, no, no), dtype=complex)
    paif = numpy.zeros((ntr, nv, no), dtype=complex)
    for i in range(ntr):
        pbaf[i] = -cc_equations.ccsd_1rdm_ba(T1f[i], T2f[i], L1intf[i], L2intf[i])
        pjif[i] = -cc_equations.ccsd_1rdm_ji(T1f[i], T2f[i], L1intf[i], L2intf[i])
        paif[i] = -cc_equations.ccsd_1rdm_ai(T1f[i], T2f[i], L1intf[i], L2intf[i], tfac=-1.0)

    return piaf, pbaf, pjif, paif


def neq_2rdm(T1f, T1b, T1i, T2f, T2b, T2i, L1f, L1b, L1i, L2f, L2b, L2i,
             D1, D2, tir, tii, ngr, ngi, gr, gi, Gr, Gi, t):

    # integrate the new L-amplitudes
    L1intf, L1intb, L1inti = quadrature.int_L1_keldysh(
        ngr, ngi, L1f, L1b, L1i, tir, tii, D1, gr, gi, Gr, Gi)
    L2intf, L2intb, L2inti = quadrature.int_L2_keldysh(
        ngr, ngi, L2f, L2b, L2i, tir, tii, D2, gr, gi, Gr, Gi)
    ntr, nv, no = T1f.shape
    assert(ntr == ngr)

    # compute response densities
    Pijab = -L2intf[t]
    Pcdab = -cc_equations.ccsd_2rdm_cdab(T1f[t], T2f[t], L1intf[t], L2intf[t])
    Pciab = -cc_equations.ccsd_2rdm_ciab(T1f[t], T2f[t], L1intf[t], L2intf[t])
    Pbcai = -cc_equations.ccsd_2rdm_bcai(T1f[t], T2f[t], L1intf[t], L2intf[t])
    Pbjai = -cc_equations.ccsd_2rdm_bjai(T1f[t], T2f[t], L1intf[t], L2intf[t])
    Pabij = -cc_equations.ccsd_2rdm_abij(T1f[t], T2f[t], L1intf[t], L2intf[t], tfac=-1.0)
    Pjkai = -cc_equations.ccsd_2rdm_jkai(T1f[t], T2f[t], L1intf[t], L2intf[t])
    Pkaij = -cc_equations.ccsd_2rdm_kaij(T1f[t], T2f[t], L1intf[t], L2intf[t])
    Pklij = -cc_equations.ccsd_2rdm_klij(T1f[t], T2f[t], L1intf[t], L2intf[t])

    return (Pcdab, Pciab, Pbcai, Pijab, Pbjai, Pabij, Pjkai, Pkaij, Pklij)
