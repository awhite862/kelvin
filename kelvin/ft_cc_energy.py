import numpy
from pyscf import lib

einsum = lib.einsum
#einsum = einsum

def ft_cc_energy(T1, T2, f, eri, g, beta, Qterm=True):
    """Return the FT-CC free-energy.

    Args:
        T1 (array): T1 amplitudes.
        T2 (array): T2 amplitudes.
        f (array): 1-electron (Fock matrix) integrals.
        eri (array): 2-electron (ERI) integrals.
        beta (float): Inverse temperature.
        Qterm (bool,optional): Include quadratic contribution?

    """
    t1_temp = T1
    if Qterm:
        t2_temp = 0.25*T2 + 0.5*einsum('yai,ybj->yabij',T1,T1)
    else:
        t2_temp = 0.25*T2

    # integrate from 0 to beta
    t1_n = einsum('y,yai->ai',g,t1_temp)
    t2_n = einsum('y,yabij->abij',g,t2_temp)

    Es1 = einsum('ai,ia->',t1_n,f)
    Es2 = einsum('abij,ijab->',t2_n,eri)

    return (Es1 + Es2) / beta

def ft_ucc_energy(T1a, T1b, T2aa, T2ab, T2bb, fa, fb, Ia, Ib, Iabab, g, beta, Qterm=True):
    """Return the FT-CC free-energy.

    Args:
        T1 (array): T1 amplitudes.
        T2 (array): T2 amplitudes.
        f (array): 1-electron (Fock matrix) integrals.
        eri (array): 2-electron (ERI) integrals.
        beta (float): Inverse temperature.
        Qterm (bool,optional): Include quadratic contribution?

    """
    t1a_temp = T1a
    t1b_temp = T1b
    if Qterm:
        t2aa_temp = 0.25*T2aa + 0.5*einsum('yai,ybj->yabij',T1a,T1a)
        t2bb_temp = 0.25*T2bb + 0.5*einsum('yai,ybj->yabij',T1b,T1b)
        t2ab_temp = T2ab + einsum('yai,ybj->yabij',T1a,T1b)
    else:
        t2aa_temp = 0.25*T2aa
        t2ab_temp = T2ab
        t2bb_temp = 0.25*T2aa

    # integrate from 0 to beta
    t1a_n = einsum('y,yai->ai',g,t1a_temp)
    t1b_n = einsum('y,yai->ai',g,t1b_temp)
    t2aa_n = einsum('y,yabij->abij',g,t2aa_temp)
    t2ab_n = einsum('y,yabij->abij',g,t2ab_temp)
    t2bb_n = einsum('y,yabij->abij',g,t2bb_temp)

    Es1 = einsum('ai,ia->',t1a_n,fa)
    Es1 += einsum('ai,ia->',t1b_n,fb)
    Es2 = einsum('abij,ijab->',t2aa_n,Ia)
    Es2 += einsum('abij,ijab->',t2ab_n,Iabab)
    Es2 += einsum('abij,ijab->',t2bb_n,Ib)

    return (Es1 + Es2) / beta

def ft_cc_energy_neq(
        T1f,T1b,T1i,T2f,T2b,T2i,
        Ff,Fb,F,eri,gr,gi,beta,Qterm=True):

    t1f_temp = T1f
    t1b_temp = T1b
    t1i_temp = T1i
    if Qterm:
        t2f_temp = 0.25*T2f + 0.5*einsum('yai,ybj->yabij',T1f,T1f)
        t2b_temp = 0.25*T2b + 0.5*einsum('yai,ybj->yabij',T1b,T1b)
        t2i_temp = 0.25*T2i + 0.5*einsum('yai,ybj->yabij',T1i,T1i)
    else:
        t2f_temp = 0.25*T2f
        t2b_temp = 0.25*T2b
        t2i_temp = 0.25*T2i

    # integrate
    E1f_n = 1.j*einsum('y,yai,yia->',gr,t1f_temp,Ff)
    E1b_n = -1.j*einsum('y,yai,yia->',gr,t1b_temp,Fb)
    t1i_n = einsum('y,yai->ai',gi,t1i_temp)
    t2f_n = einsum('y,yabij->abij',gr,t2f_temp)
    t2b_n = einsum('y,yabij->abij',gr,t2b_temp)
    t2i_n = einsum('y,yabij->abij',gi,t2i_temp)

    E1i = einsum('ai,ia->',t1i_n,F)
    Es1 = E1f_n + E1b_n + E1i
    Es2f_n = 1.j*einsum('abij,ijab->',t2f_n,eri)
    Es2b_n = -1.j*einsum('abij,ijab->',t2b_n,eri)
    E2i = einsum('abij,ijab->',t2i_n,eri)
    Es2 = Es2f_n + Es2b_n + E2i

    return (Es1 + Es2) / beta
