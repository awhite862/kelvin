import numpy
from cqcpy.cc_energy import cc_energy_d
from cqcpy.cc_energy import cc_energy_s1


def mp0(en):
    """Return the 0th order energy."""
    return en.sum()


def ump0(ea, eb):
    """Return the 0th order energy."""
    return ea.sum() + eb.sum()


def mp1(p, f, h):
    """Return the 1st order energy."""
    ec1 = numpy.tensordot(h,p,axes=([0,1],[0,1]))
    ec2 = numpy.tensordot(f,p,axes=([0,1],[0,1]))
    return 0.5*(ec1 - ec2)


def mp2(eo, ev, fvo, Ivvoo, returnT=False):
    """Return the 2nd order energy, and optionally T-amplitudes."""
    D1 = 1/(eo[:,None] - ev[None,:])
    D2 = 1/(eo[:,None,None,None] + eo[None,:,None,None]
        - ev[None,None,:,None] - ev[None,None,None,:])

    T1 = numpy.einsum('ai,ia->ai', fvo, D1)
    T2 = numpy.einsum('abij,ijab->abij', Ivvoo, D2)

    Es = cc_energy_s1(T1, fvo.transpose(1,0))
    Ed = cc_energy_d(T2, Ivvoo.transpose(2,3,0,1))

    if returnT:
        return (Es + Ed,T1,T2)
    else:
        return Es + Ed


def mp3_singles(D1, D2, F, I):
    """Return the singles contribution to the 3rd order energy."""
    dA = (-1.0)*numpy.einsum(
        'ai,ij,ja,ia,ja->',
        F.vo, F.oo, F.ov, D1, D1)
    dB = numpy.einsum(
        'ai,ba,ib,ia,ib->',
        F.vo, F.vv, F.ov, D1, D1)
    dC = numpy.einsum(
        'aibj,jb,ia,ijab,ia->',
        I.vovo, F.ov, F.ov, D2, D1)
    dD = (-1.0)*numpy.einsum(
        'ai,bjai,jb,ia,jb->',
        F.vo, I.vovo, F.ov, D1, D1)
    dE = numpy.einsum(
        'ai,bj,aibj,ia,ijab->',
        F.vo, F.vo, I.vovo, D1, D2)
    dF = (0.5)*numpy.einsum(
        'aibj,aijk,kb,ijab,kb->',
        I.vovo, I.vooo, F.ov, D2, D1)
    dG = (-0.5)*numpy.einsum(
        'aibj,aicb,jc,ijab,jc->',
        I.vovo, I.vovv, F.ov, D2, D1)
    dH = (-0.5)*numpy.einsum(
        'aibj,jk,aibk,ijab,ikab->',
        I.vovo, F.oo, I.vovo, D2, D2)
    dI = (0.5)*numpy.einsum(
        'aibj,cb,aicj,ijab,ijac->',
        I.vovo, F.vv, I.vovo, D2, D2)
    dJ = (-0.5)*numpy.einsum(
        'bj,aijk,aibk,jb,ikab->',
        F.vo, I.vooo, I.vovo, D1, D2)
    dK = (0.5)*numpy.einsum(
        'bj,aicb,aicj,jb,ijac->',
        F.vo, I.vovv, I.vovo, D1, D2)

    dtot = dA + dB + dC + dD + dE + dF + dG + dH + dI + dJ + dK

    return dtot


def mp3_doubles(D2, I):
    """Return the doubles contribution to the 3rd order energy."""
    dX = (1.0/8.0)*numpy.einsum(
        'ijab,abcd,cdij,ijab,ijcd->',
        I.oovv, I.vvvv, I.vvoo, D2, D2)
    dY = (1.0/8.0)*numpy.einsum(
        'ijab,klij,abkl,ijab,klab->',
        I.oovv, I.oooo, I.vvoo, D2, D2)
    dZ = (-1.0)*numpy.einsum(
        'ijab,bkci,ackj,ijab,kjac->',
        I.oovv, I.vovo, I.vvoo, D2, D2)

    return dX + dY + dZ


def mp3(eo, ev, F, I):
    """Return the total 3rd order contribution to the energy."""
    D1 = 1/(eo[:,None] - ev[None,:])
    D2 = 1/(eo[:,None,None,None] + eo[None,:,None,None]
        - ev[None,None,:,None] - ev[None,None,None,:])

    E3d = mp3_doubles(D2,I)
    E3s = mp3_singles(D1,D2,F,I)
    return E3d + E3s
