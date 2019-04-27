import numpy
from cqcpy.cc_energy import cc_energy_d
from cqcpy.cc_energy import cc_energy_s1

def mp0(g0):
    """Return 0th order free energy."""
    return g0.sum()

def ump0(g0a, g0b):
    """Return 0th order free energy."""
    return (g0a.sum() + g0b.sum())

def mp1(p, f, h):
    """Return 1st order free energy."""
    # compute contributions to 0th and 1st order energy
    ec1 = numpy.tensordot(h,p,axes=([0,1],[0,1])) 
    ec2 = numpy.tensordot(f,p,axes=([0,1],[0,1]))

    # compute free energy at 1st order
    return 0.5*(ec1 - ec2)

def get_D1c(e):
    """Return conventional 1-electron denominators."""
    n = e.shape[0]
    D1 = e[:,None] - e[None,:]
    for i in range(n):
        for j in range(n):
            if numpy.abs(D1[i,j]) < 1e-8:
                D1[i,j] = 1e16
    return 1.0/D1

def get_D1a(e,T):
    """Return anomalous 1-electron denominators."""
    n = e.shape[0]
    D1 = e[:,None] - e[None,:]
    for i in range(n):
        for j in range(n):
            if numpy.abs(D1[i,j]) < 1e-8:
                D1[i,j] = -2*T
            else:
                D1[i,j] = 1e16
    return 1.0/D1


def get_D2c(e):
    """Return conventional 2-electron denominators."""
    n = e.shape[0]
    D2 = e[:,None,None,None] + e[None,:,None,None] \
        - e[None,None,:,None] - e[None,None,None,:]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if numpy.abs(D2[i,j,k,l]) < 1e-8:
                        D2[i,j,k,l] = 1e32
    return 1.0/D2

def get_D2a(e,T):
    """Return anomalous 2-electron denominators."""
    n = e.shape[0]
    D2 = e[:,None,None,None] + e[None,:,None,None] \
        - e[None,None,:,None] - e[None,None,None,:]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if numpy.abs(D2[i,j,k,l]) < 1e-8:
                        D2[i,j,k,l] = -2*T
                    else:
                        D2[i,j,k,l] = 1e32
    return 1.0/D2

def get_D1exp(e,T):
    """Return 1-electron propagator."""
    beta = 1.0/T
    n = e.shape[0]
    D1 = e[:,None] - e[None,:]
    return numpy.exp(beta*D1) - 1.0

def get_D2exp(e,T):
    """Return 2-electron propagator."""
    beta = 1.0/T
    n = e.shape[0]
    D2 = e[:,None,None,None] + e[None,:,None,None] \
        - e[None,None,:,None] - e[None,None,None,:]
    return numpy.exp(beta*D2) - 1.0

def mp2(en, no, f, eri, T, returnT=False):
    """Return 2nd order correction to the free energy."""
    rpre = 1e-8
    rfac = -2.0*T
    D1 = en[:,None] - en[None,:]
    D2 = en[:,None,None,None] + en[None,:,None,None]\
        - en[None,None,:,None] - en[None,None,None,:]

    for x in numpy.nditer(D1,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac
    for x in numpy.nditer(D2,op_flags=['readwrite']):
        if numpy.abs(x) < rpre:
            x += rfac

    D1 = 1.0/D1
    D2 = 1.0/D2

    nv = (1.0 - no)
    T1 = numpy.einsum(
        'a,i,ai,ia->ai',
        nv,no,f,D1)
    T2 = numpy.einsum(
        'a,i,b,j,abij,ijab->abij',
        nv,no,nv,no,eri,D2)

    Es = cc_energy_s1(T1, f.transpose(1,0))
    Ed = cc_energy_d(T2, eri)
    
    if returnT:
        return (Es + Ed, T1, T2)
    else:
        return Es + Ed

#def ump2(ea, eb, noa, nva, nob, nvb, fa, fb, Ia, Ib, Iabab, T, returnT=False):
#    """Return unrestricted 2nd order correction to the free energy."""
#    rpre = 1e-8
#    rfac = -2.0*T
#    D1a = ea[:,None] - ea[None,:]
#    D1b = eb[:,None] - eb[None,:]
#    D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
#            - ea[None,None,:,None] - ea[None,None,None,:]
#    D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
#            - ea[None,None,:,None] - eb[None,None,None,:]
#    D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
#            - eb[None,None,:,None] - eb[None,None,None,:]
#    for x in numpy.nditer(D1a,op_flags=['readwrite']):
#        if numpy.abs(x) < rpre:
#            x += rfac
#    for x in numpy.nditer(D1b,op_flags=['readwrite']):
#        if numpy.abs(x) < rpre:
#            x += rfac
#    for x in numpy.nditer(D2aa,op_flags=['readwrite']):
#        if numpy.abs(x) < rpre:
#            x += rfac
#    for x in numpy.nditer(D2ab,op_flags=['readwrite']):
#        if numpy.abs(x) < rpre:
#            x += rfac
#    for x in numpy.nditer(D2bb,op_flags=['readwrite']):
#        if numpy.abs(x) < rpre:
#            x += rfac
#
#    D1a = 1.0/D1a
#    D1b = 1.0/D1b
#    D2aa = 1.0/D2aa
#    D2ab = 1.0/D2ab
#    D2bb = 1.0/D2bb
#
#    T1a = numpy.einsum(
#        'a,i,ai,ia->ai',
#        nva,noa,fa,D1a)
#    T1b = numpy.einsum(
#        'a,i,ai,ia->ai',
#        nvb,nob,fb,D1b)
#    T2aa = numpy.einsum('a,b,i,j,abij,ijab->abij',
#        nva,nva,noa,noa,Ia,D2)
#
#    Es = cc_energy_s1(T1, f.transpose(1,0))
#    Ed = cc_energy_d(T2, eri)
#
#    if returnT:
#        return (Es + Ed, T1, T2)
#    else:
#        return Es + Ed
    
def mp2_a(D1a, D2a, no, f, eri, T):
    """Return anomalous 2nd order correction to the free energy."""
    nv = (1.0 - no)
    T1 = numpy.einsum(
        'a,i,ai,ia->ai',
        nv,no,f,D1a)
    T2 = numpy.einsum(
        'a,i,b,j,abij,ijab->abij',
        nv,no,nv,no,eri,D2a)

    Es = cc_energy_s1(T1, f.transpose(1,0))
    Ed = cc_energy_d(T2, eri)
    
    return Es + Ed


def mp2_sep(en, no, f, eri, T, returnT=False):
    """Return 2nd order correction and compute separate MP2 
    T amplitudes.
    """
    D1 = get_D1c(en)
    D2 = get_D2c(en)
    D1a = get_D1a(en,T)
    D2a = get_D2a(en,T)

    nv = (1.0 - no)
    T1n = numpy.einsum(
        'a,i,ai,ia->ai',
        nv,no,f,D1)
    T2n = numpy.einsum(
        'a,i,b,j,abij,ijab->abij',
        nv,no,nv,no,eri,D2)
    T1a = numpy.einsum(
        'a,i,ai,ia->ai',
        nv,no,f,D1a)
    T2a = numpy.einsum(
        'a,i,b,j,abij,ijab->abij',
        nv,no,nv,no,eri,D2a)

    Es = cc_energy_s1(T1a + T1n, f.transpose(1,0))
    Ed = cc_energy_d(T2a + T2a, eri)
    
    if returnT:
        return (Es + Ed, T1n, T2n, T1a, T2a)
    else:
        return Es + Ed

def mp3_doubles(no, eri, D21, D22):
    """Return conventional doubles piece of the 3rd order 
    correction to the free energy.
    """
    nv = (1.0 - no)
    F1 = (1.0/8.0) * numpy.einsum(
        'i,j,k,l,a,b,ijab,klab,abij,ijkl,klab->',
        no,no,no,no,nv,nv,D21,D22,eri,eri,eri)
    F2 = (1.0/8.0) * numpy.einsum(
        'i,j,a,b,c,d,ijab,ijcd,abij,cdab,ijcd->',
        no,no,nv,nv,nv,nv,D21,D22,eri,eri,eri)
    F3 = -numpy.einsum(
        'i,j,k,a,b,c,ijab,kjac,ijab,kbic,ackj->',
        no,no,no,nv,nv,nv,D21,D22,eri,eri,eri)
    return F1 + F2 + F3

def mp3_doubles1N(no, eri, D21, D22):
    """Return anomalous doubles piece of the 3rd order 
    correction to the free energy.
    """
    nv = (1.0 - no)
    F1 = (1.0/8.0) * numpy.einsum(
        'i,j,k,l,a,b,ijab,klij,abij,ijkl,klab->',
        no,no,no,no,nv,nv,D21,D22,eri,eri,eri)
    F2 = (1.0/8.0) * numpy.einsum(
        'i,j,a,b,c,d,ijab,abcd,abij,cdab,ijcd->',
        no,no,nv,nv,nv,nv,D21,D22,eri,eri,eri)
    F3 = -numpy.einsum(
        'i,j,k,a,b,c,ijab,kbic,ijab,kbic,ackj->',
        no,no,no,nv,nv,nv,D21,D22,eri,eri,eri)
    return F1 + F2 + F3

def mp3_doubles2N(no, eri, D21, D22):
    """Return the doubly anomalous doubles piece of the 3rd order 
    correction to the free energy.
    """
    nv = (1.0 - no)
    F1 = (1.0/8.0) * numpy.einsum(
        'i,j,k,l,a,b,ijkl,klab,abij,ijkl,klab->',
        no,no,no,no,nv,nv,D21,D22,eri,eri,eri)
    F2 = (1.0/8.0) * numpy.einsum(
        'i,j,a,b,c,d,cdab,ijcd,abij,cdab,ijcd->',
        no,no,nv,nv,nv,nv,D21,D22,eri,eri,eri)
    F3 = -numpy.einsum(
        'i,j,k,a,b,c,ickb,kjac,ijab,kbic,ackj->',
        no,no,no,nv,nv,nv,D21,D22,eri,eri,eri)
    return F1 + F2 + F3

def mp3_singles(no, f, eri, D11, D12, D21, D22):
    """Return the conventional singles piece of the 3rd order 
    correction to the free energy.
    """
    nv = (1.0 - no)
    dA = (-1.0)*numpy.einsum(
        'i,j,a,ai,ij,ja,ia,ja->',
        no,no,nv,f,f,f,D11,D12)
    dB = numpy.einsum(
        'i,a,b,ai,ba,ib,ia,ib->',
        no,nv,nv,f,f,f,D11,D12)
    dC = numpy.einsum(
        'i,j,a,b,abij,jb,ia,ijab,ia->',
        no,no,nv,nv,eri,f,f,D21,D12)
    dD = numpy.einsum(
        'i,j,a,b,ai,bija,jb,ia,jb->',
        no,no,nv,nv,f,eri,f,D11,D12)
    dE = numpy.einsum(
        'i,j,a,b,ai,bj,ijab,ia,ijab->',
        no,no,nv,nv,f,f,eri,D11,D22)
    dF = (-0.5)*numpy.einsum(
        'i,j,k,a,b,abij,ijak,kb,ijab,kb->',
        no,no,no,nv,nv,eri,eri,f,D21,D12)
    dG = (0.5)*numpy.einsum(
        'i,j,a,b,c,abij,icab,jc,ijab,jc->',
        no,no,nv,nv,nv,eri,eri,f,D21,D12)
    dH = (-0.5)*numpy.einsum(
        'i,j,k,a,b,abij,jk,ikab,ijab,ikab->',
        no,no,no,nv,nv,eri,f,eri,D21,D22)
    dI = (0.5)*numpy.einsum(
        'i,j,a,b,c,abij,cb,ijac,ijab,ijac->',
        no,no,nv,nv,nv,eri,f,eri,D21,D22)
    dJ = (-0.5)*numpy.einsum(
        'i,j,k,a,b,bj,ajik,abik,jb,ikab->',
        no,no,no,nv,nv,f,eri,eri,D11,D22)
    dK = (0.5)*numpy.einsum(
        'i,j,a,b,c,bj,acib,ijac,jb,ijac->',
        no,no,nv,nv,nv,f,eri,eri,D11,D22)
    dtot = dA + dB + dC + dD + dE + dF + dG + dH + dI + dJ + dK

    return dtot

def mp3_singles1N(no, f, eri, D11, D12, D21, D22):
    """Return the anomalous singles piece of the 3rd order 
    correction to the free energy.
    """
    nv = (1.0 - no)
    dA = (-1.0)*numpy.einsum(
        'i,j,a,ai,ij,ja,ia,ji->',
        no,no,nv,f,f,f,D11,D12)
    dB = numpy.einsum(
        'i,a,b,ai,ba,ib,ia,ab->',
        no,nv,nv,f,f,f,D11,D12)
    dC = numpy.einsum(
        'i,j,a,b,abij,jb,ia,ijab,bj->',
        no,no,nv,nv,eri,f,f,D21,D12)
    dD = numpy.einsum(
        'i,j,a,b,ai,bija,jb,ia,jabi->',
        no,no,nv,nv,f,eri,f,D11,D22)
    dE = numpy.einsum(
        'i,j,a,b,ai,bj,ijab,ia,jb->',
        no,no,nv,nv,f,f,eri,D11,D12)
    dF = (-0.5)*numpy.einsum(
        'i,j,k,a,b,abij,ijak,kb,ijab,kaij->',
        no,no,no,nv,nv,eri,eri,f,D21,D22)
    dG = (0.5)*numpy.einsum(
        'i,j,a,b,c,abij,icab,jc,ijab,abci->',
        no,no,nv,nv,nv,eri,eri,f,D21,D22)
    dH = (-0.5)*numpy.einsum(
        'i,j,k,a,b,abij,jk,ikab,ijab,kj->',
        no,no,no,nv,nv,eri,f,eri,D21,D12)
    dI = (0.5)*numpy.einsum(
        'i,j,a,b,c,abij,cb,ijac,ijab,bc->',
        no,no,nv,nv,nv,eri,f,eri,D21,D12)
    dJ = (-0.5)*numpy.einsum(
        'i,j,k,a,b,bj,ajik,abik,jb,ikaj->',
        no,no,no,nv,nv,f,eri,eri,D11,D22)
    dK = (0.5)*numpy.einsum(
        'i,j,a,b,c,bj,acib,ijac,jb,ibac->',
        no,no,nv,nv,nv,f,eri,eri,D11,D22)
    dtot = dA + dB + dC + dD + dE + dF + dG + dH + dI + dJ + dK

    return dtot

def mp3_singles2N(no, f, eri, D11, D12, D21, D22):
    """Return the doubly anomalous singles piece of the 3rd order 
    correction to the free energy.
    """
    nv = (1.0 - no)
    dA = (-1.0)*numpy.einsum(
        'i,j,a,ai,ij,ja,ij,ja->',
        no,no,nv,f,f,f,D11,D12)
    dB = numpy.einsum(
        'i,a,b,ai,ba,ib,ba,ib->',
        no,nv,nv,f,f,f,D11,D12)
    dC = numpy.einsum(
        'i,j,a,b,abij,jb,ia,jb,ia->',
        no,no,nv,nv,eri,f,f,D11,D12)
    dD = numpy.einsum(
        'i,j,a,b,ai,bija,jb,ibaj,jb->',
        no,no,nv,nv,f,eri,f,D21,D12)
    dE = numpy.einsum(
        'i,j,a,b,ai,bj,ijab,bj,ijab->',
        no,no,nv,nv,f,f,eri,D11,D22)
    dF = (-0.5)*numpy.einsum(
        'i,j,k,a,b,abij,ijak,kb,ijak,kb->',
        no,no,no,nv,nv,eri,eri,f,D21,D12)
    dG = (0.5)*numpy.einsum(
        'i,j,a,b,c,abij,icab,jc,icab,jc->',
        no,no,nv,nv,nv,eri,eri,f,D21,D12)
    dH = (-0.5)*numpy.einsum(
        'i,j,k,a,b,abij,jk,ikab,jk,ikab->',
        no,no,no,nv,nv,eri,f,eri,D11,D22)
    dI = (0.5)*numpy.einsum(
        'i,j,a,b,c,abij,cb,ijac,cb,ijac->',
        no,no,nv,nv,nv,eri,f,eri,D11,D22)
    dJ = (-0.5)*numpy.einsum(
        'i,j,k,a,b,bj,ajik,abik,jaik,ikab->',
        no,no,no,nv,nv,f,eri,eri,D21,D22)
    dK = (0.5)*numpy.einsum(
        'i,j,a,b,c,bj,acib,ijac,acib,ijac->',
        no,no,nv,nv,nv,f,eri,eri,D21,D22)
    dtot = dA + dB + dC + dD + dE + dF + dG + dH + dI + dJ + dK

    return dtot

def mp3(e, no, f, eri, T):
    """Return the total 3rd order correction to the free energy."""
    raise Exception("This MP3 code is incorrect")
    D1 = get_D1c(e)
    D2 = get_D2c(e)
    D1a = get_D1a(e,T)
    D2a = get_D2a(e,T)

    E3d = mp3_doubles(no, eri, D2, D2)
    E3s = mp3_singles(no, f, eri, D1, D1, D2, D2)

    E3da1 = mp3_doubles(no, eri, D2a, D2)
    E3da2 = mp3_doubles(no, eri, D2, D2a)
    E3sa1 = mp3_singles(no, f, eri, D1a, D1, D2a, D2)
    E3sa2 = mp3_singles(no, f, eri, D1, D1a, D2, D2a)

    E3da12 = mp3_doubles(no, eri, D2a, D2a)
    E3sa12 = mp3_singles(no, f, eri, D1a, D1a, D2a, D2a)
    E3da12 *= (4.0)/(6.0)
    E3sa12 *= (4.0)/(6.0)

    D1 = D1**2
    D2 = D2**2
    E3daa1 = 2*T*mp3_doubles(no, eri, D2a, D2)
    E3daa2 = 2*T*mp3_doubles(no, eri, D2, D2a)
    E3saa1 = 2*T*mp3_singles(no, f, eri, D1a, D1, D2a, D2)
    E3saa2 = 2*T*mp3_singles(no, f, eri, D1, D1a, D2, D2a)
    #print(E3d,E3daa1,E3daa2,E3da12)

    return E3d + E3s + E3da1 + E3da2 + E3sa1 + E3sa2 + E3da12 + E3sa12 + \
            E3daa1 + E3daa2 + E3saa1 + E3saa2

def mp3_a(D1, D2, D1a, D2a, no, f, eri, T):
    """Return the total anomalous contribution to the 3rd order 
    correction to the free energy.
    """
    raise Exception("MP3 code is incorrect")
    E3da1 = mp3_doubles(no, eri, D2a, D2)
    E3da2 = mp3_doubles(no, eri, D2, D2a)
    E3sa1 = mp3_singles(no, f, eri, D1a, D1, D2a, D2)
    E3sa2 = mp3_singles(no, f, eri, D1, D1a, D2, D2a)

    E3da12 = mp3_doubles(no, eri, D2a, D2a)
    E3sa12 = mp3_singles(no, f, eri, D1a, D1a, D2a, D2a)
    E3da12 *= (4.0)/(6.0)
    E3sa12 *= (4.0)/(6.0)

    D1 = D1**2
    D2 = D2**2
    E3daa1 = T*mp3_doubles(no, eri, D2a, D2)
    E3daa2 = T*mp3_doubles(no, eri, D2, D2a)
    E3saa1 = T*mp3_singles(no, f, eri, D1a, D1, D2a, D2)
    E3saa2 = T*mp3_singles(no, f, eri, D1, D1a, D2, D2a)

    return E3da1 + E3da2 + E3sa1 + E3sa2 + E3da12 + E3sa12 + E3daa1 + E3daa2 + E3saa1 + E3saa2

def mp3_new(e, no, f, eri, T):
    """Return the total 3rd order correction to the free energy."""
    raise Exception("This MP3 code contains a bug somewhere")
    D1 = get_D1c(e)
    D2 = get_D2c(e)
    D1a = -2*T*get_D1a(e,T)
    D2a = -2*T*get_D2a(e,T)

    D1exp = get_D1exp(e,T)
    D2exp = get_D2exp(e,T)

    Dtemp = numpy.einsum(
        'ijab,ijab,ijab->ijab',D2,D2,D2exp)
    DN = mp3_doubles(no, eri, D2, D2)
    DN1 = T*mp3_doubles1N(no, eri, Dtemp, D2)
    DN2 = T*mp3_doubles2N(no, eri, D2, Dtemp)

    Dtemp = numpy.einsum(
        'ijab,ijab,ijab,ijab->ijab',D2,D2,D2,D2exp)
    DA1 = -(1.0/(2.0*T))*mp3_doubles(no, eri, D2a, D2)
    DA2 = -(1.0/(2.0*T))*mp3_doubles(no, eri, D2, D2a)
    DA3 = T*mp3_doubles(no, eri, D2a, Dtemp)
    DA4 = T*mp3_doubles(no, eri, Dtemp, D2a)

    DAA = 1.0/(T*T*6.0) * mp3_doubles(no, eri, D2a, D2a)

    D = DN+DN1+DN2+DA1+DA2+DA3+DA4+DAA

    Stemp = numpy.einsum(
        'ia,ia,ia->ia',D1,D1,D1exp)
    SN = mp3_singles(no, f, eri, D1, D1, D2, D2)
    SN1 = T*mp3_singles1N(no, f, eri, Stemp, D1, Dtemp, D2)
    SN2 = T*mp3_singles2N(no, f, eri, D1, Stemp, D2, Dtemp)

    Stemp = numpy.einsum(
        'ia,ia,ia,ia->ia',D1,D1,D1,D1exp)
    SA1 = -(1.0/(2.0*T))*mp3_singles(no, f, eri, D1a, D1, D2a, D2)
    SA2 = -(1.0/(2.0*T))*mp3_singles(no, f, eri, D1, D1a, D2, D2a)
    SA3 = T*mp3_singles(no, f, eri, D1a, Stemp, D2a, Dtemp)
    SA4 = T*mp3_singles(no, f, eri, Stemp, D1a, Dtemp, D2a)

    SAA = 1.0/(T*T*6.0) * mp3_singles(no, f, eri, D1a, D1a, D2a, D2a)

    S = SN+SN1+SN2+SA1+SA2+SA3+SA4+SAA
    return S+D

from . import ft_cc_energy
from . import ft_cc_equations
from . import quadrature
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks

def mp23_int(e, no, nv, f, eri, T, ngrid=10):
    """Return the 2nd and 3rd order corrections to the free energy by 
    imaginary time integration.
    """
    beta = 1.0 / (T + 1e-12)
    n = e.shape[0]

    # get time-grid
    ng = ngrid
    ti,g,G = quadrature.simpsons(ng, beta)

    # get exponentials
    D1 = e[:,None] - e[None,:]
    D2 = e[:,None,None,None] + e[None,:,None,None] \
            - e[None,None,:,None] - e[None,None,None,:]

    # get MP2 (1st order) T-amplitudes
    Id = numpy.ones((ng))
    T1old = -numpy.einsum('v,ai,a,i->vai',Id,f,nv,no)
    T2old = -numpy.einsum('v,abij,a,b,i,j->vabij',Id,eri,nv,nv,no,no)
    T1old = quadrature.int_tbar1(ng,T1old,ti,D1,G)
    T2old = quadrature.int_tbar2(ng,T2old,ti,D2,G)

    E23_1 = ft_cc_energy.ft_cc_energy(T1old,T2old,f,eri,
            g,beta)
    E23_1F = ft_cc_energy.ft_cc_energy(T1old,T2old,f,eri,
            g,beta,Qterm=False)
    E23_Q = E23_1 - E23_1F

    # pre-contract with fermi factors
    Foo = numpy.einsum('ij,j->ij',f,no)
    Fvo = numpy.einsum('ai,a,i->ai',f,nv,no)
    Fvv = numpy.einsum('ab,a->ab',f,nv)
    F = one_e_blocks(Foo,f,Fvo,Fvv)

    Ivvvv = numpy.einsum('abcd,a,b->abcd',eri,nv,nv)
    Ivvvo = numpy.einsum('abci,a,b,i->abci',eri,nv,nv,no)
    Ivovv = numpy.einsum('aibc,a->aibc',eri,nv)
    Ivvoo = numpy.einsum('abij,a,b,i,j->abij',eri,nv,nv,no,no)
    Ivovo = numpy.einsum('ajbi,a,i->ajbi',eri,nv,no)
    Ivooo = numpy.einsum('akij,a,i,j->akij',eri,nv,no,no)
    Iooov = numpy.einsum('jkia,i->jkia',eri,no)
    Ioooo = numpy.einsum('klij,i,j->klij',eri,no,no)
    I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
            vovo=Ivovo,oovv=eri,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

    # do one iteration of LCCSD
    T1,T2 = ft_cc_equations.lccsd_simple(F,I,T1old,T2old,
            D1,D2,ti,ng,G)

    E23_2 = ft_cc_energy.ft_cc_energy(T1,T2,f,eri,
            g,beta,Qterm=False)

    #print(E23_Q, E23_2)
    return E23_Q + E23_2
