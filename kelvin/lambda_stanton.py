import numpy

einsum = numpy.einsum

def _Lambda_Stanton(L1, L2, F, I, L1old, L2old, T1old, T2old, fac=1.0):
    FI_vv = F.vv - 0.5*einsum('bi,ia->ba',T1old,F.ov)
    FI_oo = F.oo - 0.5*einsum('bj,ib->ij',T1old,F.ov)

    Fov = F.ov.copy()
    Fov1 = einsum('jkbc,ck->jb',I.oovv,T1old)
    Fov += Fov1

    T2B = T2old.copy()
    T2B += einsum('ai,bj->abij',T1old,T1old)
    T2B -= einsum('bi,aj->abij',T1old,T1old)

    Woooo = I.oooo.copy()
    Woooo += einsum('klic,cj->klij',I.ooov,T1old)
    Woooo -= einsum('kljc,ci->klij',I.ooov,T1old)
    Woooo += 0.25*einsum('klcd,cdij->klij',I.oovv,T2B)
    Woooo += 0.25*einsum('abkl,abij->ijkl',T2B,I.vvoo)

    Wvvvv = I.vvvv.copy()
    Wvvvv -= einsum('akcd,bk->abcd',I.vovv,T1old)
    Wvvvv += einsum('bkcd,ak->abcd',I.vovv,T1old)
    Wvvvv += 0.25*einsum('klcd,abkl->abcd',I.oovv,T2B)
    Wvvvv += 0.25*einsum('abij,ijcd->abcd',T2B,I.oovv)

    Wovvo = -I.vovo.transpose((1,0,2,3))
    Wovvo -= einsum('bkcd,dj->kbcj',I.vovv,T1old)
    Wovvo += einsum('kljc,bl->kbcj',I.ooov,T1old)
    temp = 0.5*T2old + einsum('dj,bl->dbjl',T1old,T1old)
    Wovvo -= einsum('klcd,dbjl->kbcj',I.oovv,temp)
    Wovvo += 0.5*einsum('acjk,kicb->iabj',T2old,I.oovv)

    Wtemp = -I.vovo.transpose((1,0,2,3))
    Wtemp -= einsum('bckj,ikac->ibaj',T2old,I.oovv)

    #temp = einsum('bj,iabj->iajk',T1old,Wtemp)
    temp = einsum('bj,iabk->iajk',T1old,Wtemp)
    temp += einsum('iljb,abkl->iajk',I.ooov,T2old)
    temp -= temp.transpose((0,1,3,2))
    Wovoo = -I.vooo.transpose((1,0,2,3)) - einsum('ib,abjk->iajk',Fov,T2old)
    Wovoo -= einsum('al,iljk->iajk',T1old,Woooo)
    Wovoo -= 0.5*einsum('aibc,bcjk->iajk',I.vovv,T2B)
    Wovoo += temp

    temp = einsum('bjad,cdij->bcai',I.vovv,T2old)
    temp -= einsum('bj,jcai->bcai',T1old,Wtemp)
    temp -= temp.transpose((1,0,2,3))
    Wvvvo = I.vvvo + einsum('ja,bcij->bcai',Fov,T2old)
    Wvvvo += einsum('di,bcad->bcai',T1old,Wvvvv)
    Wvvvo += 0.5*einsum('aijk,bcjk->bcai',I.vooo,T2B)
    Wvvvo += temp

    LToo = 0.5*einsum('abjk,ikab->ji',T2old,L2old)
    LTvv = -0.5*einsum('bcij,ijac->ab',T2old,L2old)

    L1 += Fov1 + einsum('ib,ba->ia',L1old, FI_vv)
    L1 -= einsum('ja,ij->ia',L1old, FI_oo)
    L1 += 0.5*einsum('ijbc,bcaj->ia',L2old, Wvvvo)
    L1 += einsum('me,eima->ia',L1old, Wovvo.transpose((1,0,3,2))) # fix this
    #L1 -= 0.5*einsum('mnae,mnie->ia',L2old,Wooov)
    L1 -= einsum('ef,eifa->ia',LTvv,I.vovv)
    L1 -= einsum('mn,mina->ia',LToo,I.ooov)
    Temp1 = einsum('fe,fm->em',LTvv,T1old)
    Temp1 -= einsum('mn,en->em',LToo,T1old)
    L1 += einsum('em,imae->ia',Temp1,I.oovv)

    #L2 = I.oovv.copy()
    #L2 +=

    return L1,L2
