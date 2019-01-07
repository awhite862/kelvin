import time
import numpy
from pyscf import lib
from cqcpy.ov_blocks import one_e_blocks
from cqcpy import cc_equations
from cqcpy import ft_utils
from . import quadrature

einsum = lib.einsum
#einsum = einsum

def lccd_simple(F,I,T2old,D2,ti,ng,G):
    """Time-dependent linearized coupled cluster 
    doubles (LCCD) iteration.
    """
    t1 = time.time()

    Id = numpy.ones((ng))
    T2new = -einsum('v,abij->vabij',Id,I.vvoo)

    for y in range(ng):
        cc_equations._D_D(T2new[y,:,:,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)

    # integrate the new T-amplitudes
    T2new = quadrature.int_tbar2(ng,T2new,ti,D2,G)

    return T2new

def lccsd_simple(F,I,T1old,T2old,D1,D2,ti,ng,G):
    """Time-dependent linearized coupled cluster 
    singles and doubles (LCCSD) iteration.
    """
    t1 = time.time()

    Id = numpy.ones((ng))
    T1new = -einsum('v,ai->vai',Id,F.vo)
    T2new = -einsum('v,abij->vabij',Id,I.vvoo)

    for y in range(ng):
        cc_equations._S_S(T1new[y,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._S_D(T1new[y,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._D_S(T2new[y,:,:,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._D_D(T2new[y,:,:,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)

    # integrate the new T-amplitudes
    T1new = quadrature.int_tbar1(ng,T1new,ti,D1,G)
    T2new = quadrature.int_tbar2(ng,T2new,ti,D2,G)

    return T1new,T2new

#def lccsd_rt(f,eri,T1old,T2old,Dvo,Dvvoo,ti,ng,delta):
#
#    t1 = time.time()
#
#    # form the new T-amplitudes by contraction
#    Id = numpy.ones(ng)
#    T1new = -numpy.einsum('v,ai->vai',Id,f.vo)
#    T2new = -numpy.einsum('v,abij->vabij',Id,eri.vvoo)
#
#    for y in range(ng):
#        cc_equations._S_S(T1new[y,:,:],f,eri,T1old[y,:,:],fac=-1.0)
#        cc_equations._S_D(T1new[y,:,:],f,eri,T2old[y,:,:,:,:],fac=-1.0)
#        cc_equations._D_S(T2new[y,:,:,:,:],f,eri,T1old[y,:,:],fac=-1.0)
#        cc_equations._D_D(T2new[y,:,:,:,:],f,eri,T2old[y,:,:,:,:],fac=-1.0)
#
#    # integrate the new T-amplitudes
#    G = quadrature.get_G(ng,delta)
#    T1new = quadrature.int_tbar1(ng,T1new,ti,Dvo,G)
#    T2new = quadrature.int_tbar2(ng,T2new,ti,Dvvoo,G)
#
#    return T1new,T2new

def ccd_simple(F,I,T2old,D2,ti,ng,G):
    """Time-dependent coupled cluster doubles (CCD) 
    iteration.
    """
    t1 = time.time()

    Id = numpy.ones((ng))
    T2new = -einsum('v,abij->vabij',Id,I.vvoo)

    for y in range(ng):
        cc_equations._D_D(T2new[y,:,:,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._D_DD(T2new[y,:,:,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)

    # integrate the new T-amplitudes
    T2new = quadrature.int_tbar2(ng,T2new,ti,D2,G)

    return T2new

def ccsd_simple(F,I,T1old,T2old,D1,D2,ti,ng,G):
    """Time-dependent coupled cluster singles and 
    doubles (CCSD) iteration.
    """
    t1 = time.time()

    Id = numpy.ones((ng))
    T1new = -einsum('v,ai->vai',Id,F.vo)
    T2new = -einsum('v,abij->vabij',Id,I.vvoo)

    for y in range(ng):
        cc_equations._S_S(T1new[y,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._S_D(T1new[y,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._S_SS(T1new[y,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._S_SD(T1new[y,:,:],F,I,T1old[y,:,:],T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._S_SSS(T1new[y,:,:],F,I,T1old[y,:,:],fac=-1.0)

        cc_equations._D_S(T2new[y,:,:,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._D_D(T2new[y,:,:,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._D_SS(T2new[y,:,:,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._D_SD(T2new[y,:,:,:,:],F,I,T1old[y,:,:],T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._D_DD(T2new[y,:,:,:,:],F,I,T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._D_SSD(T2new[y,:,:,:,:],F,I,T1old[y,:,:],T2old[y,:,:,:,:],fac=-1.0)
        cc_equations._D_SSS(T2new[y,:,:,:,:],F,I,T1old[y,:,:],fac=-1.0)
        cc_equations._D_SSSS(T2new[y,:,:,:,:],F,I,T1old[y,:,:],fac=-1.0)

    # integrate the new T-amplitudes
    T1new = quadrature.int_tbar1(ng,T1new,ti,D1,G)
    T2new = quadrature.int_tbar2(ng,T2new,ti,D2,G)

    return T1new,T2new

def ccsd_stanton(F,I,T1old,T2old,D1,D2,ti,ng,G):
    """Time-dependent coupled cluster singles and 
    doubles (CCSD) iteration using Stanton-Gauss 
    intermediates.
    """
    t1 = time.time()

    Id = numpy.ones((ng))
    T1new = -einsum('v,ai->vai',Id,F.vo)
    T2new = -einsum('v,abij->vabij',Id,I.vvoo)

    for y in range(ng):
        cc_equations._Stanton(T1new[y,:,:],T2new[y,:,:,:,:],
                F,I,T1old[y,:,:],T2old[y,:,:,:,:],fac=-1.0)

    # integrate the new T-amplitudes
    T1new = quadrature.int_tbar1(ng,T1new,ti,D1,G)
    T2new = quadrature.int_tbar2(ng,T2new,ti,D2,G)

    return T1new,T2new

def uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,T2aaold,T2abold,T2bbold,
        D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G):
    """Time-dependent coupled cluster singles and
    doubles (CCSD) iteration using Stanton-Gauss
    intermediates.
    """
    t1 = time.time()
    na = D1a.shape[0]
    nb = D1b.shape[0]

    #Id = numpy.ones((ng))
    T1a = numpy.zeros((ng,na,na))
    T1b = numpy.zeros((ng,nb,nb))
    T2aa = numpy.zeros((ng,na,na,na,na))
    T2ab = numpy.zeros((ng,na,nb,na,nb))
    T2bb = numpy.zeros((ng,nb,nb,nb,nb))

    for y in range(ng):
        T1a[y,:,:] = -Fa.vo.copy()
        T1b[y,:,:] = -Fb.vo.copy()
        T2aa[y,:,:,:,:] = -Ia.vvoo.copy()
        T2bb[y,:,:,:,:] = -Ib.vvoo.copy()
        T2ab[y,:,:,:,:] = -Iabab.vvoo.copy()
        T1olds = (T1aold[y,:,:],T1bold[y,:,:])
        T2olds = (T2aaold[y,:,:,:,:],T2abold[y,:,:,:,:],T2bbold[y,:,:,:,:])
        cc_equations._u_Stanton(T1a[y,:,:], T1b[y,:,:], T2aa[y,:,:,:,:], T2ab[y,:,:,:,:], T2bb[y,:,:,:,:],
                Fa, Fb, Ia, Ib, Iabab, T1olds, T2olds, fac=-1.0)

    # integrate the new T-amplitudes
    T1a = quadrature.int_tbar1(ng,T1a,ti,D1a,G)
    T1b = quadrature.int_tbar1(ng,T1b,ti,D1b,G)
    T2aa = quadrature.int_tbar2(ng,T2aa,ti,D2aa,G)
    T2ab = quadrature.int_tbar2(ng,T2ab,ti,D2ab,G)
    T2bb = quadrature.int_tbar2(ng,T2bb,ti,D2bb,G)

    return (T1a,T1b),(T2aa,T2ab,T2bb)

def neq_ccsd_simple(Ff,Fb,F,I,T1oldf,T1oldb,T1oldi,
        T2oldf,T2oldb,T2oldi,D1,D2,tir,tii,ngr,ngi,Gr,Gi):

    T1newf = -Ff.vo.copy()
    T1newb = -Fb.vo.copy()
    Idr = numpy.ones((ngr))
    Idi = numpy.ones((ngi))
    T1newi = -einsum('v,ai->vai',Idi,F.vo)
    T2newb = -einsum('v,abij->vabij',Idr,I.vvoo)
    T2newf = T2newb.copy()
    T2newi = -einsum('v,abij->vabij',Idi,I.vvoo)

    for y in range(ngr):
        Fftemp = one_e_blocks(Ff.oo[y],Ff.ov[y],Ff.vo[y],Ff.vv[y])
        Fbtemp = one_e_blocks(Fb.oo[y],Fb.ov[y],Fb.vo[y],Fb.vv[y])
        cc_equations._S_S(T1newf[y],Fftemp,I,T1oldf[y],fac=-1.0)
        cc_equations._S_D(T1newf[y],Fftemp,I,T2oldf[y],fac=-1.0)
        cc_equations._S_SS(T1newf[y],Fftemp,I,T1oldf[y],fac=-1.0)
        cc_equations._S_SD(T1newf[y],Fftemp,I,T1oldf[y],T2oldf[y],fac=-1.0)
        cc_equations._S_SSS(T1newf[y],Fftemp,I,T1oldf[y],fac=-1.0)

        cc_equations._D_S(T2newf[y],Fftemp,I,T1oldf[y],fac=-1.0)
        cc_equations._D_D(T2newf[y],Fftemp,I,T2oldf[y],fac=-1.0)
        cc_equations._D_SS(T2newf[y],Fftemp,I,T1oldf[y],fac=-1.0)
        cc_equations._D_SD(T2newf[y],Fftemp,I,T1oldf[y],T2oldf[y],fac=-1.0)
        cc_equations._D_DD(T2newf[y],Fftemp,I,T2oldf[y],fac=-1.0)
        cc_equations._D_SSD(T2newf[y],Fftemp,I,T1oldf[y],T2oldf[y],fac=-1.0)
        cc_equations._D_SSS(T2newf[y],Fftemp,I,T1oldf[y],fac=-1.0)
        cc_equations._D_SSSS(T2newf[y],Fftemp,I,T1oldf[y],fac=-1.0)

        cc_equations._S_S(T1newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)
        cc_equations._S_D(T1newb[y],Fbtemp,I,T2oldb[y],fac=-1.0)
        cc_equations._S_SS(T1newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)
        cc_equations._S_SD(T1newb[y],Fbtemp,I,T1oldb[y],T2oldb[y],fac=-1.0)
        cc_equations._S_SSS(T1newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)

        cc_equations._D_S(T2newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)
        cc_equations._D_D(T2newb[y],Fbtemp,I,T2oldb[y],fac=-1.0)
        cc_equations._D_SS(T2newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)
        cc_equations._D_SD(T2newb[y],Fbtemp,I,T1oldb[y],T2oldb[y],fac=-1.0)
        cc_equations._D_DD(T2newb[y],Fbtemp,I,T2oldb[y],fac=-1.0)
        cc_equations._D_SSD(T2newb[y],Fbtemp,I,T1oldb[y],T2oldb[y],fac=-1.0)
        cc_equations._D_SSS(T2newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)
        cc_equations._D_SSSS(T2newb[y],Fbtemp,I,T1oldb[y],fac=-1.0)

    for y in range(ngi):
        cc_equations._S_S(T1newi[y],F,I,T1oldi[y],fac=-1.0)
        cc_equations._S_D(T1newi[y],F,I,T2oldi[y],fac=-1.0)
        cc_equations._S_SS(T1newi[y],F,I,T1oldi[y],fac=-1.0)
        cc_equations._S_SD(T1newi[y],F,I,T1oldi[y],T2oldi[y],fac=-1.0)
        cc_equations._S_SSS(T1newi[y],F,I,T1oldi[y],fac=-1.0)

        cc_equations._D_S(T2newi[y],F,I,T1oldi[y],fac=-1.0)
        cc_equations._D_D(T2newi[y],F,I,T2oldi[y],fac=-1.0)
        cc_equations._D_SS(T2newi[y],F,I,T1oldi[y],fac=-1.0)
        cc_equations._D_SD(T2newi[y],F,I,T1oldi[y],T2oldi[y],fac=-1.0)
        cc_equations._D_DD(T2newi[y],F,I,T2oldi[y],fac=-1.0)
        cc_equations._D_SSD(T2newi[y],F,I,T1oldi[y],T2oldi[y],fac=-1.0)
        cc_equations._D_SSS(T2newi[y],F,I,T1oldi[y],fac=-1.0)
        cc_equations._D_SSSS(T2newi[y],F,I,T1oldi[y],fac=-1.0)

    T1newf,T1newb,T1newi = quadrature.int_tbar1_keldysh(
        ngr,ngi,T1newf,T1newb,T1newi,tir,tii,D1,Gr,Gi)
    T2newf,T2newb,T2newi = quadrature.int_tbar2_keldysh(
        ngr,ngi,T2newf,T2newb,T2newi,tir,tii,D2,Gr,Gi)

    return T1newf,T1newb,T1newi,T2newf,T2newb,T2newi

def neq_ccsd_stanton(Ff,Fb,F,I,T1oldf,T1oldb,T1oldi,
        T2oldf,T2oldb,T2oldi,D1,D2,tir,tii,ngr,ngi,Gr,Gi):

    T1newf = -Ff.vo.copy()
    T1newb = -Fb.vo.copy()
    Idr = numpy.ones((ngr))
    Idi = numpy.ones((ngi))
    T1newi = -einsum('v,ai->vai',Idi,F.vo)
    T2newb = -einsum('v,abij->vabij',Idr,I.vvoo)
    T2newf = T2newb.copy()
    T2newi = -einsum('v,abij->vabij',Idi,I.vvoo)

    for y in range(ngr):
        Fftemp = one_e_blocks(Ff.oo[y],Ff.ov[y],Ff.vo[y],Ff.vv[y])
        Fbtemp = one_e_blocks(Fb.oo[y],Fb.ov[y],Fb.vo[y],Fb.vv[y])
        cc_equations._Stanton(T1newf[y],T2newf[y],Fftemp,I,T1oldf[y],T2oldf[y],fac=-1.0)
        cc_equations._Stanton(T1newb[y],T2newb[y],Fbtemp,I,T1oldb[y],T2oldb[y],fac=-1.0)
    for y in range(ngi):
        cc_equations._Stanton(T1newi[y],T2newi[y],F,I,T1oldi[y],T2oldi[y],fac=-1.0)

    T1newf,T1newb,T1newi = quadrature.int_tbar1_keldysh(
        ngr,ngi,T1newf,T1newb,T1newi,tir,tii,D1,Gr,Gi)
    T2newf,T2newb,T2newi = quadrature.int_tbar2_keldysh(
        ngr,ngi,T2newf,T2newb,T2newi,tir,tii,D2,Gr,Gi)

    return T1newf,T1newb,T1newi,T2newf,T2newb,T2newi

def lccd_lambda_simple(F,I,T2old,L2old,D2,ti,ng,g,G,beta):
    """Time-dependent linearized coupled cluster doubles (LCCD) 
    Lambda iteration. 
    """
    t1 = time.time()
    # integrate old lambda amplitudes
    L2int = quadrature.int_L2(ng,L2old,ti,D2,g,G)

    # initialize lambda amplitudes
    L2 = numpy.zeros(L2old.shape)

    # amplitude term
    for y in range(ng):
        cc_equations._LD_LD(L2[y,:,:,:,:], F, I, L2int[y,:,:,:,:], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L2 += (1.0/beta)*einsum('v,ijab->vijab',Id,I.oovv)
    t2 = time.time()

    return L2

def lccsd_lambda_simple(F,I,T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G,beta):
    """Time-dependent linearized coupled cluster singles and doubles (LCCSD) 
    Lambda iteration. 
    """
    t1 = time.time()
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng,L1old,ti,D1,g,G)
    L2int = quadrature.int_L2(ng,L2old,ti,D2,g,G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape)
    L2 = numpy.zeros(L2old.shape)

    # amplitude term
    for y in range(ng):
        cc_equations._LS_LS(L1[y,:,:], F, I, L1int[y,:,:], fac=-1.0)
        cc_equations._LS_LD(L1[y,:,:], F, I, L2int[y,:,:,:,:], fac=-1.0)

        cc_equations._LD_LS(L2[y,:,:,:,:], F, I, L1int[y,:,:], fac=-1.0)
        cc_equations._LD_LD(L2[y,:,:,:,:], F, I, L2int[y,:,:,:,:], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 += einsum('v,ia->via',Id,F.ov)
    L2 += einsum('v,ijab->vijab',Id,I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y,:,:],I,T1old[y,:,:],fac=1.0)
    t2 = time.time()

    return L1,L2

def ccsd_lambda_simple(F,I,T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G,beta):
    """Time-dependent coupled cluster singles and doubles (CCSD) 
    Lambda iteration. 
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng,L1old,ti,D1,g,G)
    L2int = quadrature.int_L2(ng,L2old,ti,D2,g,G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape)
    L2 = numpy.zeros(L2old.shape)

    # amplitude term
    for y in range(ng):
        cc_equations._LS_LS(L1[y,:,:], F, I, L1int[y,:,:], fac=-1.0)
        cc_equations._LS_LSTS(L1[y,:,:], F, I, L1int[y,:,:], T1old[y,:,:], fac=-1.0)
        cc_equations._LS_LSTD(L1[y,:,:], I, L1int[y,:,:], T2old[y,:,:,:,:], fac=-1.0)
        cc_equations._LS_LSTSS(L1[y,:,:], I, L1int[y,:,:], T1old[y,:,:], fac=-1.0)
        cc_equations._LS_LD(L1[y,:,:], F, I, L2int[y,:,:,:,:], fac=-1.0)
        cc_equations._LS_LDTS(L1[y,:,:], F, I, L2int[y,:,:,:,:], T1old[y,:,:], fac=-1.0)
        cc_equations._LS_LDTD(L1[y,:,:], F, I, L2int[y,:,:,:,:], T2old[y,:,:,:,:], fac=-1.0)
        cc_equations._LS_LDTSS(L1[y,:,:], F, I, L2int[y,:,:,:,:], T1old[y,:,:], fac=-1.0)
        cc_equations._LS_LDTSD(L1[y,:,:], I, L2int[y,:,:,:,:], T1old[y,:,:], T2old[y,:,:,:,:], fac=-1.0)
        cc_equations._LS_LDTSSS(L1[y,:,:], I, L2int[y,:,:,:,:], T1old[y,:,:], fac=-1.0)

        cc_equations._LD_LS(L2[y,:,:,:,:], F, I, L1int[y,:,:], fac=-1.0)
        cc_equations._LD_LSTS(L2[y,:,:,:,:], F, I, L1int[y,:,:], T1old[y,:,:], fac=-1.0)
        cc_equations._LD_LD(L2[y,:,:,:,:], F, I, L2int[y,:,:,:,:], fac=-1.0)
        cc_equations._LD_LDTS(L2[y,:,:,:,:], F, I, L2int[y,:,:,:,:], T1old[y,:,:], fac=-1.0)
        cc_equations._LD_LDTD(L2[y,:,:,:,:], I, L2int[y,:,:,:,:], T2old[y,:,:,:,:], fac=-1.0)
        cc_equations._LD_LDTSS(L2[y,:,:,:,:], F, I, L2int[y,:,:,:,:], T1old[y,:,:], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 += einsum('v,ia->via',Id,F.ov)
    L2 += einsum('v,ijab->vijab',Id,I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y,:,:],I,T1old[y,:,:],fac=1.0)

    return L1,L2

def ccsd_lambda_opt(F,I,T1old,T2old,L1old,L2old,D1,D2,ti,ng,g,G,beta):
    """Time-dependent coupled cluster singles and doubles (CCSD) 
    Lambda iteration with intermediates.
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng,L1old,ti,D1,g,G)
    L2int = quadrature.int_L2(ng,L2old,ti,D2,g,G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape)
    L2 = numpy.zeros(L2old.shape)

    # amplitude term
    t1 = time.time()
    for y in range(ng):
        cc_equations._Lambda_opt(L1[y,:,:], L2[y,:,:,:,:], F, I,
                L1int[y,:,:], L2int[y,:,:,:,:], T1old[y,:,:], T2old[y,:,:,:,:], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 += einsum('v,ia->via',Id,F.ov)
    L2 += einsum('v,ijab->vijab',Id,I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y,:,:],I,T1old[y,:,:],fac=1.0)

    return L1,L2

def uccsd_lambda_opt(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,T2aaold,T2abold,T2bbold,
        L1aold,L1bold,L2aaold,L2abold,L2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,g,G,beta):
    """Time-dependent coupled cluster singles and doubles (CCSD)
    Lambda iteration with intermediates.
    """
    na = D1a.shape[0]
    nb = D1b.shape[0]
    n = na + nb

    # integrate old lambda amplitudes
    L1aint = quadrature.int_L1(ng,L1aold,ti,D1a,g,G)
    L1bint = quadrature.int_L1(ng,L1bold,ti,D1b,g,G)
    L2aaint = quadrature.int_L2(ng,L2aaold,ti,D2aa,g,G)
    L2abint = quadrature.int_L2(ng,L2abold,ti,D2ab,g,G)
    L2bbint = quadrature.int_L2(ng,L2bbold,ti,D2bb,g,G)

    # initialize lambda amplitudes
    L1a = numpy.zeros((ng,na,na))
    L1b = numpy.zeros((ng,nb,nb))
    L2aa = numpy.zeros((ng,na,na,na,na))
    L2ab = numpy.zeros((ng,na,nb,na,nb))
    L2bb = numpy.zeros((ng,nb,nb,nb,nb))

    # amplitude term
    t1 = time.time()
    for y in range(ng):
        L1olds = (L1aint[y],L1bint[y])
        T1olds = (T1aold[y],T1bold[y])
        L2olds = (L2aaint[y],L2abint[y],L2bbint[y])
        T2olds = (T2aaold[y],T2abold[y],T2bbold[y])
        cc_equations._uccsd_Lambda_opt(L1a[y,:,:], L1b[y,:,:], L2aa[y,:,:,:,:],
                L2ab[y,:,:,:,:], L2bb[y,:,:,:,:], Fa, Fb, Ia, Ib, Iabab,
                L1olds, L2olds, T1olds, T2olds, fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1a += einsum('v,ia->via',Id,Fa.ov)
    L1b += einsum('v,ia->via',Id,Fb.ov)
    L2aa += einsum('v,ijab->vijab',Id,Ia.oovv)
    L2ab += einsum('v,ijab->vijab',Id,Iabab.oovv)
    L2bb += einsum('v,ijab->vijab',Id,Ib.oovv)
    for y in range(ng):
        T1olds = (T1aold[y,:,:],T1bold[y,:,:])
        cc_equations._u_LS_TS(L1a[y,:,:],L1b[y,:,:],Ia,Ib,Iabab,T1olds[0],T1olds[1])

    return L1a,L1b,L2aa,L2ab,L2bb

def ft_ccsd_lambda_int(F, I, T1old, T2old):
    ng = T1old.shape[0]
    intor = []
    for y in range(ng):
        intor.append(cc_equations.lambda_int(F, I, T1old[y,:,:], T2old[y,:,:,:,:]))
    return intor 

def ccsd_lambda_opt_int(F,I,T1old,T2old,L1old,L2old,intor,D1,D2,ti,ng,g,G,beta):
    """Time-dependent coupled cluster singles and doubles (CCSD) 
    Lambda iteration with precomputed intermediates.
    """
    # integrate old lambda amplitudes
    L1int = quadrature.int_L1(ng,L1old,ti,D1,g,G)
    L2int = quadrature.int_L2(ng,L2old,ti,D2,g,G)

    # initialize lambda amplitudes
    L1 = numpy.zeros(L1old.shape)
    L2 = numpy.zeros(L2old.shape)

    # amplitude term
    t1 = time.time()
    for y in range(ng):
        cc_equations._Lambda_opt_int(L1[y,:,:], L2[y,:,:,:,:], F, I,
            L1int[y,:,:], L2int[y,:,:,:,:], T1old[y,:,:], T2old[y,:,:,:,:], intor[y], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L1 += einsum('v,ia->via',Id,F.ov)
    L2 += einsum('v,ijab->vijab',Id,I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y,:,:],I,T1old[y,:,:],fac=1.0)


    return L1,L2

def ccsd_lambda_guess(F,I,T1old,T2old,beta):
    """Time-dependent coupled cluster singles and doubles (CCSD) 
    Lambda guess.
    """
    ng = T2old.shape[0]
    Id = numpy.ones((ng))
    L1 = (1.0/beta)*einsum('v,ia->via',Id,F.ov)
    L2 = (1.0/beta)*einsum('v,ijab->vijab',Id,I.oovv)
    for y in range(ng):
        cc_equations._LS_TS(L1[y,:,:],I,T1old[y,:,:],fac=(1.0/beta))

    return L1,L2

def neq_lambda_simple(Ff,Fb,F,I,L1oldf,L1oldb,L1oldi,
        L2oldf,L2oldb,L2oldi,T1oldf,T1oldb,T1oldi,
        T2oldf,T2oldb,T2oldi,D1,D2,tir,tii,ngr,ngi,gr,gi,Gr,Gi):

    # integrate old lambda amplitudes
    L1intf,L1intb,L1inti = quadrature.int_L1_keldysh(
            ngr,ngi,L1oldf,L1oldb,L1oldi,tir,tii,D1,gr,gi,Gr,Gi)
    L2intf,L2intb,L2inti = quadrature.int_L2_keldysh(
            ngr,ngi,L2oldf,L2oldb,L2oldi,tir,tii,D2,gr,gi,Gr,Gi)

    # initialize lambda amplitudes
    L1f = numpy.zeros(L1oldf.shape,dtype=complex)
    L1b = numpy.zeros(L1oldb.shape,dtype=complex)
    L1i = numpy.zeros(L1oldi.shape,dtype=complex)
    L2f = numpy.zeros(L2oldf.shape,dtype=complex)
    L2b = numpy.zeros(L2oldb.shape,dtype=complex)
    L2i = numpy.zeros(L2oldi.shape,dtype=complex)
    #print(L2intf[0])

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
        Fbtemp = one_e_blocks(Fb.oo[y],Fb.ov[y],Fb.vo[y],Fb.vv[y])
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

        Fftemp = one_e_blocks(Ff.oo[y],Ff.ov[y],Ff.vo[y],Ff.vv[y])
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

        cc_equations._LS_TS(L1f[y],I,T1oldf[y],fac=1.0)
        cc_equations._LS_TS(L1b[y],I,T1oldb[y],fac=1.0)

    for y in range(ngi):
        L1i[y] += F.ov
        L2i[y] += I.oovv
        cc_equations._LS_TS(L1i[y],I,T1oldi[y],fac=1.0)

    return L1f,L1b,L1i,L2f,L2b,L2i

def neq_lambda_opt(Ff,Fb,F,I,L1oldf,L1oldb,L1oldi,
        L2oldf,L2oldb,L2oldi,T1oldf,T1oldb,T1oldi,
        T2oldf,T2oldb,T2oldi,D1,D2,tir,tii,ngr,ngi,gr,gi,Gr,Gi):

    # integrate old lambda amplitudes
    L1intf,L1intb,L1inti = quadrature.int_L1_keldysh(
            ngr,ngi,L1oldf,L1oldb,L1oldi,tir,tii,D1,gr,gi,Gr,Gi)
    L2intf,L2intb,L2inti = quadrature.int_L2_keldysh(
            ngr,ngi,L2oldf,L2oldb,L2oldi,tir,tii,D2,gr,gi,Gr,Gi)

    # initialize lambda amplitudes
    L1f = numpy.zeros(L1oldf.shape,dtype=complex)
    L1b = numpy.zeros(L1oldb.shape,dtype=complex)
    L1i = numpy.zeros(L1oldi.shape,dtype=complex)
    L2f = numpy.zeros(L2oldf.shape,dtype=complex)
    L2b = numpy.zeros(L2oldb.shape,dtype=complex)
    L2i = numpy.zeros(L2oldi.shape,dtype=complex)

    # amplitude term
    for y in range(ngr):
        Fftemp = one_e_blocks(Ff.oo[y],Ff.ov[y],Ff.vo[y],Ff.vv[y])
        cc_equations._Lambda_opt(L1f[y], L2f[y], Fftemp, I,
                L1intf[y], L2intf[y], T1oldf[y], T2oldf[y], fac=-1.0)
        Fbtemp = one_e_blocks(Fb.oo[y],Fb.ov[y],Fb.vo[y],Fb.vv[y])
        cc_equations._Lambda_opt(L1b[y], L2b[y], Fbtemp, I,
                L1intb[y], L2intb[y], T1oldb[y], T2oldb[y], fac=-1.0)

    for y in range(ngi):
        cc_equations._Lambda_opt(L1i[y], L2i[y], F, I,
                L1inti[y], L2inti[y], T1oldi[y], T2oldi[y], fac=-1.0)

    # energy term
    for y in range(ngr):
        L1f[y] += Ff.ov[y]
        L1b[y] += Fb.ov[y]
        L2f[y] += I.oovv
        L2b[y] += I.oovv

        cc_equations._LS_TS(L1f[y],I,T1oldf[y],fac=1.0)
        cc_equations._LS_TS(L1b[y],I,T1oldb[y],fac=1.0)

    for y in range(ngi):
        L1i[y] += F.ov
        L2i[y] += I.oovv
        cc_equations._LS_TS(L1i[y],I,T1oldi[y],fac=1.0)

    return L1f,L1b,L1i,L2f,L2b,L2i


def ccd_lambda_simple(F,I,T2old,L2old,D2,ti,ng,g,G,beta):
    """Time-dependent coupled cluster doubles (CCD) 
    Lambda iteration.
    """
    t1 = time.time()
    # integrate old lambda amplitudes
    L2int = quadrature.int_L2(ng,L2old,ti,D2,g,G)

    # initialize lambda amplitudes
    L2 = numpy.zeros(L2old.shape)

    # amplitude term
    for y in range(ng):
        cc_equations._LD_LD(L2[y,:,:,:,:], F, I, L2int[y,:,:,:,:], fac=-1.0)
        cc_equations._LD_LDTD(L2[y,:,:,:,:], I, L2int[y,:,:,:,:], T2old[y,:,:,:,:], fac=-1.0)

    # energy term
    Id = numpy.ones((ng))
    L2 += einsum('v,ijab->vijab',Id,I.oovv)
    #L2 += (1.0/beta)*numpy.einsum('v,ijab->vijab',Id,I.oovv)
    t2 = time.time()

    return L2

def ccsd_1rdm(T1,T2,L1,L2,D1,D2,ti,ng,delta):

    # integrate the new L-amplitudes
    G = quadrature.get_G(ng,delta)
    g = quadrature.get_gint(ng,delta)
    L1new = quadrature.int_L1(ng,L1,ti,D1,g,G)
    L2new = quadrature.int_L2(ng,L2,ti,D2,g,G)
    nt,nv,no = T1.shape
    assert(nt == ng)

    # compute response densities
    pia = -numpy.einsum('sia,s->ia',L1new,g)
    pba = numpy.zeros((nv,nv))
    pji = numpy.zeros((no,no))
    pai = numpy.zeros((nv,no))
    for i in range(nt):
        pba += g[i]*cc_equations.ccsd_1rdm_ba(T1[i],T2[i],L1new[i],L2new[i])
        pji += g[i]*cc_equations.ccsd_1rdm_ji(T1[i],T2[i],L1new[i],L2new[i])
        pai += g[i]*cc_equations.ccsd_1rdm_ai(T1[i],T2[i],L1new[i],L2new[i])

    return pia,pba,pji,pai

def ccsd_2rdm(T1,T2,L1,L2,D1,D2,ti,ng,delta):
    nt,nv,no = T1.shape
    assert(nt == ng)

    # integrate the new L-amplitudes
    G = quadrature.get_G(ng,delta)
    g = quadrature.get_gint(ng,delta)
    L1new = quadrature.int_L1(ng,L1,ti,D1,g,G)
    L2new = quadrature.int_L2(ng,L2,ti,D2,g,G)

    # compute response densities
    Pcdab = numpy.zeros((nv,nv,nv,nv))
    Pciab = numpy.zeros((nv,no,nv,nv))
    Pbcai = numpy.zeros((nv,nv,nv,no))
    Pijab = -einsum('sijab,s->ijab',L2new,g)
    Pbjai = numpy.zeros((nv,no,nv,no))
    Pabij = numpy.zeros((nv,nv,no,no))
    Pjkai = numpy.zeros((no,no,nv,no))
    Pkaij = numpy.zeros((no,nv,no,no))
    Pklij = numpy.zeros((no,no,no,no))
    for i in range(nt):
        Pcdab += g[i]*cc_equations.ccsd_2rdm_cdab(T1[i],T2[i],L1new[i],L2new[i])
        Pciab += g[i]*cc_equations.ccsd_2rdm_ciab(T1[i],T2[i],L1new[i],L2new[i])
        Pbcai += g[i]*cc_equations.ccsd_2rdm_bcai(T1[i],T2[i],L1new[i],L2new[i])
        Pbjai += g[i]*cc_equations.ccsd_2rdm_bjai(T1[i],T2[i],L1new[i],L2new[i])
        Pabij += g[i]*cc_equations.ccsd_2rdm_abij(T1[i],T2[i],L1new[i],L2new[i])
        Pjkai += g[i]*cc_equations.ccsd_2rdm_jkai(T1[i],T2[i],L1new[i],L2new[i])
        Pkaij += g[i]*cc_equations.ccsd_2rdm_kaij(T1[i],T2[i],L1new[i],L2new[i])
        Pklij += g[i]*cc_equations.ccsd_2rdm_klij(T1[i],T2[i],L1new[i],L2new[i])

    return (Pcdab, Pciab, Pbcai, Pijab, Pbjai, Pabij, Pjkai, Pkaij, Pklij)

def neq_1rdm(T1f,T1b,T1i,T2f,T2b,T2i,L1f,L1b,L1i,L2f,L2b,L2i,
        D1,D2,tir,tii,ngr,ngi,gr,gi,Gr,Gi):

    # integrate the new L-amplitudes
    L1intf,L1intb,L1inti = quadrature.int_L1_keldysh(ngr,ngi,L1f,L1b,L1i,
            tir,tii,D1,gr,gi,Gr,Gi)
    L2intf,L2intb,L2inti = quadrature.int_L2_keldysh(ngr,ngi,L2f,L2b,L2i,
            tir,tii,D2,gr,gi,Gr,Gi)
    ntr,nv,no = T1f.shape
    assert(ntr == ngr)

    # compute response densities
    piaf = -einsum('y,yia->yia',gr,L1intf)
    pbaf = numpy.zeros((ntr,nv,nv),dtype=complex)
    pjif = numpy.zeros((ntr,no,no),dtype=complex)
    paif = numpy.zeros((ntr,nv,no),dtype=complex)
    for i in range(ntr):
        pbaf[i] = gr[i]*cc_equations.ccsd_1rdm_ba(T1f[i],T2f[i],L1intf[i],L2intf[i])
        pjif[i] = gr[i]*cc_equations.ccsd_1rdm_ji(T1f[i],T2f[i],L1intf[i],L2intf[i])
        paif[i] = gr[i]*cc_equations.ccsd_1rdm_ai(T1f[i],T2f[i],L1intf[i],L2intf[i])

    return piaf,pbaf,pjif,paif

def neq_2rdm(T1,T2,L1,L2,D1,D2,ti,tir,tii,ngr,ngi,gr,gi,Gr,Gi,ng,delta,t):
    nt,nv,no = T1.shape
    assert(nt == ng)

    # integrate the new L-amplitudes
    L1intf,L1intb,L1inti = quadrature.int_L1_keldysh(ngr,ngi,L1f,L1b,L1i,
            tir,tii,D1,gr,gi,Gr,Gi)
    L2intf,L2intb,L2inti = quadrature.int_L2_keldysh(ngr,ngi,L2f,L2b,L2i,
            tir,tii,D2,gr,gi,Gr,Gi)
    ntr,nv,no = T1f.shape
    assert(ntr == ngr)

    # compute response densities
    Pijab = -gr[t]*L2new
    Pcdab = gr[t]*cc_equations.ccsd_2rdm_cdab(T1[t],T2[t],L1new[t],L2new[t])
    Pciab = gr[t]*cc_equations.ccsd_2rdm_ciab(T1[t],T2[t],L1new[t],L2new[t])
    Pbcai = gr[t]*cc_equations.ccsd_2rdm_bcai(T1[t],T2[t],L1new[t],L2new[t])
    Pbjai = gr[t]*cc_equations.ccsd_2rdm_bjai(T1[t],T2[t],L1new[t],L2new[t])
    Pabij = gr[t]*cc_equations.ccsd_2rdm_abij(T1[t],T2[t],L1new[t],L2new[t])
    Pjkai = gr[t]*cc_equations.ccsd_2rdm_jkai(T1[t],T2[t],L1new[t],L2new[t])
    Pkaij = ge[t]*cc_equations.ccsd_2rdm_kaij(T1[t],T2[t],L1new[i],L2new[t])
    Pklij = ge[t]*cc_equations.ccsd_2rdm_klij(T1[t],T2[t],L1new[i],L2new[t])

    return (Pcdab, Pciab, Pbcai, Pijab, Pbjai, Pabij, Pjkai, Pkaij, Pklij)
