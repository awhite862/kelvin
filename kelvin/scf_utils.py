import numpy
from pyscf import gto, scf
from cqcpy import utils

def get_r_orbital_energies(mf):
    """Get restricted orbital energies in o-v blocks."""

    mo_occ = mf.mo_occ
    if len(mo_occ.shape) == 1:
        eo = mf.mo_energy[mo_occ>0]
        ev = mf.mo_energy[mo_occ==0]
        return (eo, ev)
    else:
        raise Exception("Mean-field object is not restricted")

def get_u_orbital_energies(mf):
    """Get unrestricted orbital energies in o-v blocks."""

    mo_occ = mf.mo_occ
    if len(mo_occ.shape) == 1:
        eo = mf.mo_energy[mo_occ>0]
        ev = mf.mo_energy[mo_occ==0]
        return (eo, ev, eo, ev)
    elif len(mo_occ.shape) == 2:
        eoa = mf.mo_energy[0][mo_occ[0]>0]
        eva = mf.mo_energy[0][mo_occ[0]==0]
        eob = mf.mo_energy[1][mo_occ[1]>0]
        evb = mf.mo_energy[1][mo_occ[1]==0]
        return (eoa, eva, eob, evb)
    else:
        raise Exception("Unexpected size of mo_coeffs")

def get_orbital_energies(mf):
    """Get spin-orbital orbital energies in o-v blocks."""
    mo_occ = mf.mo_occ

    if len(mo_occ.shape) == 1:
        eo = mf.mo_energy[mo_occ>0]
        ev = mf.mo_energy[mo_occ==0]
        eo2 = numpy.concatenate((eo,eo))
        ev2 = numpy.concatenate((ev,ev))
        return (eo2, ev2)

    elif len(mo_occ.shape) == 2:
        mo_occa = mf.mo_occ[0]
        mo_occb = mf.mo_occ[1]
        eoa = (mf.mo_energy[0])[mo_occa>0]
        eva = (mf.mo_energy[0])[mo_occa==0]
        eob = (mf.mo_energy[1])[mo_occb>0]
        evb = (mf.mo_energy[1])[mo_occb==0]
        eo = numpy.concatenate((eoa,eob))
        ev = numpy.concatenate((eva,evb))
        return (eo,ev)

    else:
        raise Exception("unrecognized SCF type")

def get_r_orbital_energies_tot(mf):
    """Get all restricted orbital energies."""
    if len(mf.mo_occ.shape) == 1:
        return mf.mo_energy
    else:
        raise Exception("Mean-field object is not restricted")

def get_u_orbital_energies_tot(mf):
    """Get all restricted orbital energies."""
    if len(mf.mo_occ.shape) == 1:
        return (mf.mo_energy,mf.mo_energy)
    elif len(mf.mo_occ.shape) == 2:
        return mf.mo_energy[0],mf.mo_energy[1]
    else:
        raise Exception("Unexpected size of mo_energy")

def get_orbital_energies_gen(mf):
    """Get all spin-orbital orbital energies."""
    mo_occ = mf.mo_occ

    if len(mo_occ.shape) == 1:
        e = mf.mo_energy
        return numpy.concatenate((e,e))

    elif len(mo_occ.shape) == 2:
        ea = mf.mo_energy[0]
        eb = mf.mo_energy[1]
        return numpy.concatenate((ea,eb))

    else:
        raise Exception("unrecognized SCF type")


def get_ao_den(mf):
    if len(mf.mo_occ.shape) == 1:
        o = mf.mo_coeff[:,mf.mo_occ>0]
        p = numpy.dot(o,o.T)
        return utils.block_diag(p,p)

    elif len(mf.mo_occ.shape) == 2:
        mo_occa = mf.mo_occ[0]
        mo_occb = mf.mo_occ[1]
        oa = (mf.mo_coeff[0])[:,mo_occa>0]
        ob = (mf.mo_coeff[1])[:,mo_occb>0]
        pa = numpy.dot(oa, oa.T)
        pb = numpy.dot(ob, ob.T)
        return utils.block_diag(pa,pb)

    else:
        raise Exception("unrecognized SCF type")

def get_ao_ft_den(mf, fo):
    if len(mf.mo_occ.shape) == 1:
        n = fo.shape[0]/2
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(fo[:n])),mo.T)
        return utils.block_diag(p,p)

    elif len(mf.mo_occ.shape) == 2:
        n = fo.shape[0]/2
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        pa = numpy.dot(numpy.dot(moa,numpy.diag(fo[:n])),moa.T)
        pb = numpy.dot(numpy.dot(mob,numpy.diag(fo[n:])),mob.T)
        return utils.block_diag(pa,pb)

    else:
        raise Exception("unrecognized SCF type")

def get_ao_fock(mf):
    if len(mf.mo_occ.shape) == 1:
        f = mf.get_fock()
        return utils.block_diag(f,f)

    elif len(mf.mo_occ.shape) == 2:
        mo_occa = mf.mo_occ[0]
        mo_occb = mf.mo_occ[1]
        oa = (mf.mo_coeff[0])[:,mo_occa>0]
        ob = (mf.mo_coeff[1])[:,mo_occb>0]
        pa = numpy.dot(oa, oa.T)
        pb = numpy.dot(ob, ob.T)
        dm = numpy.array((pa,pb))
        h1 = mf.get_hcore(mf.mol)
        pbc = False
        try:
            ktemp = mf.kpt
            pbc = True
        except AttributeError:
            pbc = False
        if pbc:
            veff = mf.get_veff(mf.cell, dm)
        else:
            veff = mf.get_veff(mf.mol, dm)
        f = (h1 + veff[0], h1 + veff[1])
        return utils.block_diag(f[0], f[1])

    else:
        raise Exception("unrecognized SCF type")

def mo_tran_1e(mf, h):
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        nao = mo.shape[0]
        assert(h.shape[0] == nao)
        hmo = numpy.einsum('mp,mn,nq->pq',mo,h,mo)
        return utils.block_diag(hmo,hmo)

    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        ha = numpy.einsum('mp,mn,nq->pq',moa,h,moa)
        hb = numpy.einsum('mp,mn,nq->pq',mob,h,mob)
        return utils.block_diag(ha,hb)

def u_mo_tran_1e(mf, h):
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        nao = mo.shape[0]
        assert(h.shape[0] == nao)
        hmo = numpy.einsum('mp,mn,nq->pq',mo,h,mo)
        return hmo,hmo

    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        ha = numpy.einsum('mp,mn,nq->pq',moa,h,moa)
        hb = numpy.einsum('mp,mn,nq->pq',mob,h,mob)
        return ha,hb

def get_ao_ft_fock(mf, fo):
    f0 = get_ao_fock(mf)
    n = fo.shape[0]/2
    pbc = False
    try:
        ktemp = mf.kpt
        pbc = True
    except AttributeError:
        pbc = False
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(fo[:n])),mo.T)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        fT = h1 + 2*veff
        return utils.block_diag(fT,fT)

    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        pa = numpy.dot(numpy.dot(moa,numpy.diag(fo[:n])),moa.T)
        pb = numpy.dot(numpy.dot(mob,numpy.diag(fo[n:])),mob.T)
        d = (pa,pb)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        #veff = mf.get_veff(mf.mol,d)
        fTa = h1 + veff[0]
        fTb = h1 + veff[1]
        return utils.block_diag(fTa,fTb)

    else:
        raise Exception("unrecognized SCF type")

def get_r_ft_fock(mf, fo):
    n = fo.shape[0]
    pbc = False
    try:
        ktemp = mf.kpt
        pbc = True
    except AttributeError:
        pbc = False
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(fo[:n])),mo.T)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        fT = h1 + 2*veff
        fmo = numpy.einsum('mp,mn,nq->pq',mo,fT,mo)
        return fmo 
    else:
        raise Exception("SCF is not resstricted")

def get_u_ft_fock(mf, foa, fob):
    pbc = False
    try:
        ktemp = mf.kpt
        pbc = True
    except AttributeError:
        pbc = False
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(foa)),mo.T)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        fT = h1 + 2*veff
        fmo = numpy.einsum('mp,mn,nq->pq',mo,fT,mo)
        return fmo,fmo

    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        pa = numpy.dot(numpy.dot(moa,numpy.diag(foa)),moa.T)
        pb = numpy.dot(numpy.dot(mob,numpy.diag(fob)),mob.T)
        d = (pa,pb)
        if pbc:
            veff = mf.get_veff(mf.cell,d)
        else:
            veff = mf.get_veff(mf.mol,d)
        fTa = h1 + veff[0]
        fTb = h1 + veff[1]
        fmoa = numpy.einsum('mp,mn,nq->pq',moa,fTa,moa)
        fmob = numpy.einsum('mp,mn,nq->pq',mob,fTb,mob)
        return fmoa,fmob

    else:
        raise Exception("unrecognized SCF type")


def get_mo_ft_fock(mf, fo):
    n = fo.shape[0]/2
    pbc = False
    try:
        ktemp = mf.kpt
        pbc = True
    except AttributeError:
        pbc = False
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(fo[:n])),mo.T)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        fT = h1 + 2*veff
        fmo = numpy.einsum('mp,mn,nq->pq',mo,fT,mo)
        return utils.block_diag(fmo,fmo)
        
    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        pa = numpy.dot(numpy.dot(moa,numpy.diag(fo[:n])),moa.T)
        pb = numpy.dot(numpy.dot(mob,numpy.diag(fo[n:])),mob.T)
        d = (pa,pb)
        if pbc:
            veff = mf.get_veff(mf.cell,d)
        else:
            veff = mf.get_veff(mf.mol,d)
        fTa = h1 + veff[0]
        fTb = h1 + veff[1]
        fmoa = numpy.einsum('mp,mn,nq->pq',moa,fTa,moa)
        fmob = numpy.einsum('mp,mn,nq->pq',mob,fTb,mob)
        return utils.block_diag(fmoa,fmob)

    else:
        raise Exception("unrecognized SCF type")

def get_mo_d_ft_fock(mf, fo, fv, dvec):
    n = fo.shape[0]/2
    fov = dvec*fo*fv
    pbc = False
    try:
        ktemp = mf.kpt
        pbc = True
    except AttributeError:
        pbc = False
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(fov[:n])),mo.T)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        fT = 2*veff
        fmo = numpy.einsum('mp,mn,nq->pq',mo,fT,mo)
        return utils.block_diag(fmo,fmo)
 
    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        pa = numpy.dot(numpy.dot(moa,numpy.diag(fov[:n])),moa.T)
        pb = numpy.dot(numpy.dot(mob,numpy.diag(fov[n:])),mob.T)
        d = (pa,pb)
        if pbc:
            veff = mf.get_veff(mf.cell,d)
        else:
            veff = mf.get_veff(mf.mol,d)
        fTa = veff[0]
        fTb = veff[1]
        fmoa = numpy.einsum('mp,mn,nq->pq',moa,fTa,moa)
        fmob = numpy.einsum('mp,mn,nq->pq',mob,fTb,mob)
        return utils.block_diag(fmoa,fmob)

    else:
        raise Exception("unrecognized SCF type")

def u_mo_d_ft_fock(mf, foa, fva, fob, fvb, dveca, dvecb):
    na = foa.shape[0]
    nb = fob.shape[0]
    fova = dveca*foa*fva
    fovb = dvecb*fob*fvb
    pbc = False
    try:
        ktemp = mf.kpt
        pbc = True
    except AttributeError:
        pbc = False
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        p = numpy.dot(numpy.dot(mo,numpy.diag(fova)),mo.T)
        h1 = mf.get_hcore(mf.mol)
        if pbc:
            veff = mf.get_veff(mf.cell,p)
        else:
            veff = mf.get_veff(mf.mol,p)
        fT = 2*veff
        fmo = numpy.einsum('mp,mn,nq->pq',mo,fT,mo)
        return fmo,fmo

    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        pa = numpy.dot(numpy.dot(moa,numpy.diag(fova)),moa.T)
        pb = numpy.dot(numpy.dot(mob,numpy.diag(fovb)),mob.T)
        d = (pa,pb)
        if pbc:
            veff = mf.get_veff(mf.cell,d)
        else:
            veff = mf.get_veff(mf.mol,d)
        fTa = veff[0]
        fTb = veff[1]
        fmoa = numpy.einsum('mp,mn,nq->pq',moa,fTa,moa)
        fmob = numpy.einsum('mp,mn,nq->pq',mob,fTb,mob)
        return fmoa,fmob
    else:
        raise Exception("unrecognized SCF type")

class r_fock_blocks(object):
    def __init__(self,mf,orb='a'):
        mo_occ = mf.mo_occ
        if len(mo_occ.shape) == 1:
            o = mf.mo_coeff[:,mo_occ>0]
            v = mf.mo_coeff[:,mo_occ==0]
            f = mf.get_fock()
            self._transform_fock(mf,o,v,f)
        elif orb == 'a':
            o = mf.mo_coeff[0][:,mo_occ[0]>0]
            v = mf.mo_coeff[0][:,mo_occ[0]==0]
            f = mf.get_fock()[0]
            self._transform_fock(mf,o,v,f)
        elif orb == 'b':
            o = mf.mo_coeff[1][:,mo_occ[1]>0]
            v = mf.mo_coeff[1][:,mo_occ[1]==0]
            f = mf.get_fock()[1]
            self._transform_fock(mf,o,v,f)
        else:
            raise Exception("SCF is not restricted")

    def _transform_fock(self, mf, o, v, f):
        self.oo = numpy.einsum('mo,mn,np->op',o,f,o)
        self.ov = numpy.einsum('mo,mn,nv->ov',o,f,v)
        self.vo = numpy.einsum('mv,mn,no->vo',v,f,o)
        self.vv = numpy.einsum('mv,mn,nu->vu',v,f,v)

class g_fock_blocks(object):
    def __init__(self,mf):
        mo_occ = mf.mo_occ
        pbc = False
        try:
            ktemp = mf.kpt
            pbc = True
        except AttributeError:
            pbc = False
        if len(mo_occ.shape) == 1:
            o = mf.mo_coeff[:,mo_occ>0]
            v = mf.mo_coeff[:,mo_occ==0]
            f = mf.get_fock()
            self._transform_fock(mf,o,o,v,v,f,f)
        elif len(mo_occ.shape) == 2:
            mo_occa = mf.mo_occ[0]
            mo_occb = mf.mo_occ[1]
            oa = (mf.mo_coeff[0])[:,mo_occa>0]
            va = (mf.mo_coeff[0])[:,mo_occa==0]
            ob = (mf.mo_coeff[1])[:,mo_occb>0]
            vb = (mf.mo_coeff[1])[:,mo_occb==0]
            pa = numpy.dot(oa, oa.T)
            pb = numpy.dot(ob, ob.T)
            dm = numpy.array((pa,pb))
            h1 = mf.get_hcore(mf.mol)
            if pbc:
                veff = mf.get_veff(mf.cell, dm)
            else:
                veff = mf.get_veff(mf.mol, dm)
            f = (h1 + veff[0], h1 + veff[1])
            self._transform_fock(mf,oa,ob,va,vb,f[0],f[1])
        else:
            raise Exception("unrecognized SCF type")

    def _transform_fock(self, mf, oa, ob, va, vb, fa, fb):
        fvoa = numpy.einsum('mv,mn,no->vo',va,fa,oa)
        fova = numpy.einsum('mv,mn,no->vo',oa,fa,va)
        fooa = numpy.einsum('mo,mn,np->op',oa,fa,oa)
        fvva = numpy.einsum('mv,mn,nu->vu',va,fa,va)
        fvob = numpy.einsum('mv,mn,no->vo',vb,fb,ob)
        fovb = numpy.einsum('mv,mn,no->vo',ob,fb,vb)
        foob = numpy.einsum('mo,mn,np->op',ob,fb,ob)
        fvvb = numpy.einsum('mv,mn,nu->vu',vb,fb,vb)
        self.oo = utils.block_diag(fooa,foob)
        self.ov = utils.block_diag(fova,fovb)
        self.vo = utils.block_diag(fvoa,fvob)
        self.vv = utils.block_diag(fvva,fvvb)

