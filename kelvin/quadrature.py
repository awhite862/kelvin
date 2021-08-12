import numpy
from pyscf import lib
einsum = lib.einsum
#einsum = numpy.einsum


def integrate1a(t1, n, ng, delta):
    """Integrate a function of t' from 0 to t."""
    t1_temp = numpy.zeros(t1.shape)
    t1_temp[1] = 0.5*delta*(t1[0] + t1[1])
    for y in range(2, ng):
        t1_temp[y] = t1_temp[y - 2] + (delta/3.0)*(t1[y - 2]
                + 4.0*t1[y - 1] + t1[y])
    return t1_temp


def get_G_midpoint(ng, delta):
    G = numpy.zeros((ng, ng))
    G[0, 0] = delta
    for y in range(1, ng):
        G[y] = G[y - 1]
        G[y, y] += delta
    return G


def get_g_midpoint(ng, delta):
    g = numpy.zeros(ng)
    g.fill(delta)
    return g


def get_G(ng, delta):
    """Return 0 to t quadrature weight tensor for uniform grid."""
    G = numpy.zeros((ng, ng))
    G[1, 0] = G[1, 1] = 0.5*delta
    for y in range(2, ng):
        G[y] = G[y - 2]
        G[y, y - 2] += delta/3.0
        G[y, y - 1] += 4.0*delta/3.0
        G[y, y] += delta/3.0

    return G


def get_gint(ng, delta):
    """Return 0 to beta quadrature weight tensor for uniform grid."""
    g = numpy.zeros(ng)
    if ng % 2 == 0:
        g[0] += 0.5*delta
        g[1] += 0.5*delta
        for y in range(3, ng, 2):
            g[y - 2] += delta/3.0
            g[y - 1] += 4.0*delta/3.0
            g[y] += delta / 3.0
    else:
        for y in range(2, ng, 2):
            g[y - 2] += delta/3.0
            g[y - 1] += 4.0*delta/3.0
            g[y] += delta / 3.0
    return g


def get_gL(ng, delta):
    g = numpy.zeros(ng)
    for i in range(ng - 1):
        g[i] = delta
    return g


def get_GL(ng, delta):
    G = numpy.zeros((ng, ng))
    for i in range(ng):
        for j in range(i - 1):
            G[i, j] = delta
    return G


def left(ng, beta):
    delta = beta/(ng - 1.)
    ti = numpy.asarray([float(i)*delta for i in range(ng)])
    g = get_gL(ng, delta)
    G = get_GL(ng, delta)
    return ti, g, G


def d_left(ng, beta):
    delta = beta/(ng - 1.0)
    ddelta = delta/beta
    Gd = get_GL(ng, ddelta)
    gd = get_gL(ng, ddelta)
    return gd, Gd


def midpoint(ng, beta):
    delta = beta/ng
    ti = numpy.asarray([float(i)*delta + delta/2 for i in range(ng)])
    G = get_G_midpoint(ng, delta)
    g = get_g_midpoint(ng, delta)
    return ti, g, G


def simpsons(ng, beta):
    delta = beta/(ng - 1.0)
    ti = numpy.asarray([float(i)*delta for i in range(ng)])
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    return ti, g, G


def d_simpsons(ng, beta):
    delta = beta/(ng - 1.0)
    ddelta = delta/beta
    Gd = get_G(ng, ddelta)
    gd = get_gint(ng, ddelta)
    return gd, Gd


def simpsons_ln(ng, beta):
    e = numpy.exp(1)
    delta = (e - 1.0)/(ng - 1.0)
    si = numpy.asarray([(float(i)*delta + 1.0) for i in range(ng)])
    ti = beta*numpy.log(si)
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = beta*g/si
    for i in range(ng):
        G[i] = beta*G[i]/si
    return ti, g, G


def d_simpsons_ln(ng, beta):
    e = numpy.exp(1)
    delta = (e - 1.0)/(ng - 1.0)
    si = numpy.asarray([(float(i)*delta + 1.0) for i in range(ng)])
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = g/si
    for i in range(ng):
        G[i] = G[i]/si

    return g, G


def simpsons_sin(ng, beta):
    delta = numpy.pi/(ng - 1.0)
    si = numpy.asarray([(float(i)*delta - numpy.pi/2) for i in range(ng)])
    ti = beta*(numpy.sin(si) + 1.0)/2.0
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = beta*g*numpy.cos(si)/2.0
    for i in range(ng):
        G[i] = beta*G[i]*numpy.cos(si)/2.0
    return ti, g, G


def d_simpsons_sin(ng, beta):
    delta = numpy.pi/(ng - 1.0)
    si = numpy.asarray([(float(i)*delta - numpy.pi/2) for i in range(ng)])
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = g*numpy.cos(si)/2.0
    for i in range(ng):
        G[i] = G[i]*numpy.cos(si)/2.0
    return g, G


def simpsons_exp(ng, beta):
    ln2 = numpy.log(2.0)
    delta = ln2/(ng - 1.0)
    si = numpy.asarray([float(i)*delta for i in range(ng)])
    ti = beta*(numpy.exp(si) - 1.0)
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = beta*g*numpy.exp(si)
    for i in range(ng):
        G[i] = beta*G[i]*numpy.exp(si)
    return ti, g, G


def d_simpsons_exp(ng, beta):
    ln2 = numpy.log(2.0)
    delta = ln2/(ng - 1.0)
    si = numpy.asarray([float(i)*delta for i in range(ng)])
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = g*numpy.exp(si)
    for i in range(ng):
        G[i] = G[i]*numpy.exp(si)
    return g, G


def simpsons_p(ng, beta, n=2):
    delta = 1.0/(ng - 1.0)
    si = numpy.asarray([float(i)*delta for i in range(ng)])
    ti = beta*(numpy.power(si, n) + si)/2.0
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = beta*g*(numpy.power(si, n - 1)*float(n) + 1.0)/2.0
    for i in range(ng):
        G[i] = beta*G[i]*(numpy.power(si, n - 1)*float(n) + 1.0)/2.0
    return ti, g, G


def d_simpsons_p(ng, beta, n=2):
    delta = 1.0/(ng - 1.0)
    si = numpy.asarray([float(i)*delta for i in range(ng)])
    g = get_gint(ng, delta)
    G = get_G(ng, delta)
    g = g*(numpy.power(si, n - 1)*float(n) + 1.0)/2.0
    for i in range(ng):
        G[i] = G[i]*(numpy.power(si, n - 1)*float(n) + 1.0)/2.0
    return g, G


def ft_quad(ng, beta, quad):
    if quad == 'lin':
        return simpsons(ng, beta)
    elif quad == 'ln':
        return simpsons_ln(ng, beta)
    elif quad == 'sin':
        return simpsons_sin(ng, beta)
    elif quad == 'exp':
        return simpsons_exp(ng, beta)
    elif quad == 'quad':
        return simpsons_p(ng, beta, 2)
    elif quad == 'cub':
        return simpsons_p(ng, beta, 3)
    elif quad == 'quar':
        return simpsons_p(ng, beta, 4)
    elif quad == 'mid':
        return midpoint(ng, beta)
    elif quad == 'L':
        return left(ng, beta)
    else:
        raise Exception("Unrecognized quadrature rule: {}".format(quad))


def d_ft_quad(ng, beta, quad):
    if quad == 'lin':
        return d_simpsons(ng, beta)
    elif quad == 'ln':
        return d_simpsons_ln(ng, beta)
    elif quad == 'sin':
        return d_simpsons_sin(ng, beta)
    elif quad == 'exp':
        return d_simpsons_exp(ng, beta)
    elif quad == 'quad':
        return d_simpsons_p(ng, beta, 2)
    elif quad == 'cub':
        return d_simpsons_p(ng, beta, 3)
    elif quad == 'quar':
        return d_simpsons_p(ng, beta, 4)
    elif quad == 'L':
        return d_left(ng, beta)
    else:
        raise Exception("Unrecognized quadrature rule: {}".format(quad))


#def integrate_new(T,G,ng):
#    shape = T.shape
#    T = T.reshape((ng,-1))
#    temp = (einsum('yx,xp->yp',G,T)).reshape(shape)
#    T = T.reshape(shape)
#    return temp


#def integrate1(t1,n,ng,delta):
#    t1_temp = numpy.zeros(t1.shape)
#    t1_temp[1,:,:] = 0.5*delta*(t1[0,:,:] + t1[1,:,:])
#    for y in range(2,ng):
#        t1_temp[y,:,:] = t1_temp[y - 2,:,:] + (delta/3.0)*(t1[y - 2,:,:]
#                + 4.0*t1[y - 1,:,:] + t1[y,:,:])
#    return t1_temp
#
#def integrate1b(t1,n,ng,delta):
#    t1_temp = numpy.zeros(t1.shape)
#    t1_temp[1,:,:,:] = 0.5*delta*(t1[0,:,:,:] + t1[1,:,:,:])
#    for y in range(2,ng):
#        t1_temp[y,:,:,:] = t1_temp[y - 2,:,:,:] + (delta/3.0)*(t1[y - 2,:,:,:]
#                + 4.0*t1[y - 1,:,:,:] + t1[y,:,:,:])
#    return t1_temp
#
#def integrate2(t2,n,ng,delta):
#    t2_temp = numpy.zeros(t2.shape)
#    t2_temp[1,:,:,:,:] = 0.5*delta*(t2[0,:,:,:,:] + t2[1,:,:,:,:])
#    for y in range(2,ng):
#       t2_temp[y,:,:,:,:] = t2_temp[y-2,:,:,:,:] + (delta/3.0)*(t2[y - 2,:,:,:,:]
#            + 4.0*t2[y - 1,:,:,:,:] + t2[y,:,:,:,:])
#    return t2_temp


def int_tbar1(ng, t1bar, ti, D1, G):
    """Integrate t1bar with exponential factor."""
    t1_out = numpy.zeros(t1bar.shape, dtype=t1bar.dtype)
    dt = numpy.zeros((ng))
    for y in range(ng):
        for i in range(y):
            dt[i] = ti[i] - ti[y]
        gtemp = numpy.exp(dt[:, None, None]*D1[None, :, :])
        t1_temp = gtemp*t1bar
        t1_out[y] = einsum('x,xai->ai', G[y], t1_temp)

    return t1_out


def int_tbar2(ng, t2bar, ti, D2, G):
    """Integrate t2bar with exponential factor."""
    t2_out = numpy.zeros(t2bar.shape, dtype=t2bar.dtype)
    dt = numpy.zeros((ng))
    for y in range(ng):
        for i in range(y):
            dt[i] = ti[i] - ti[y]
        gtemp = numpy.exp((dt[:, None, None, None, None]*D2[None, :, :, :, :]))
        t2_temp = gtemp*t2bar
        t2_out[y] = einsum('x,xabij->abij', G[y], t2_temp)

    return t2_out


def int_L1(ng, L1old, ti, D1, g, G):
    """Return L1bar."""
    L1_out = numpy.zeros(L1old.shape, dtype=L1old.dtype)
    for s in range(ng):
        dt = numpy.zeros((ng))
        for y in range(s, ng):
            dt[y] = ti[s] - ti[y]
        gtemp = numpy.exp(dt[:, None, None]*D1[None, :, :])
        L1temp = einsum('yai,yia->yia', gtemp, L1old)
        L1_out[s] = einsum('y,yia,y->ia', g, L1temp, G[:, s])/g[s]

    return L1_out


def int_L2(ng, L2old, ti, D2, g, G):
    """Return L2bar."""
    L2_out = numpy.zeros(L2old.shape, dtype=L2old.dtype)
    for s in range(ng):
        dt = numpy.zeros((ng))
        for y in range(s, ng):
            dt[y] = ti[s] - ti[y]
        gtemp = numpy.exp(dt[:, None, None, None, None]*D2[None, :, :, :, :])
        L2temp = einsum('yabij,yijab->yijab', gtemp, L2old)
        L2_out[s] = einsum('y,yijab,y->ijab', g, L2temp, G[:, s])/g[s]

    return L2_out


def int_tbar1_single(ng, ig, t1bar, ti, D1, G):
    """Integrate t1bar with exponential factor."""
    t1_out = numpy.zeros(t1bar.shape)
    dt = numpy.zeros((ng))
    for i in range(ig):
        dt[i] = ti[i] - ti[ig]
    gtemp = numpy.exp(dt[:, None, None]*D1[None, :, :])
    t1_temp = gtemp*t1bar
    t1_out = einsum('x,xai->ai', G[ig], t1_temp)

    return t1_out


def int_tbar2_single(ng, ig, t2bar, ti, D2, G):
    """Integrate t1bar with exponential factor."""
    t2_out = numpy.zeros(t2bar.shape)
    dt = numpy.zeros((ng))
    for i in range(ig):
        dt[i] = ti[i] - ti[ig]
    gtemp = numpy.exp(dt[:, None, None, None, None]*D2[None, :, :, :, :])
    t2_temp = gtemp*t2bar
    t2_out = einsum('x,xabij->abij', G[ig], t2_temp)

    return t2_out


def int_tbar1_keldysh(ngr, ngi, t1barf, t1barb, t1bari, tir, tii, D1, Gr, Gi):
    """Integrate t1bar with exponential factor."""
    t1_outf = numpy.zeros(t1barf.shape, dtype=complex)
    dt = numpy.zeros((ngr))
    for y in range(ngr):
        for i in range(ngr):
            dt[i] = tir[i] - tir[y]
        gtemp = numpy.exp(1.j*dt[:, None, None]*D1[None, :, :])
        t1_temp = gtemp*t1barf
        t1_outf[y] = 1.j*einsum('x,xai->ai', Gr[y], t1_temp)

    t1_outb = numpy.zeros(t1barb.shape, dtype=complex)
    dt = numpy.zeros((ngr))
    tib = tir.copy()
    for i in range(ngr):
        tib[ngr - i - 1] = tir[i]
    for y in range(ngr):
        for i in range(ngr):
            dt[i] = tib[i] - tib[y]
        gtemp = numpy.exp(1.j*dt[:, None, None]*D1[None, :, :])
        t1_temp = gtemp*t1barb
        t1_outb[y] = -1.j*einsum('x,xai->ai', Gr[y], t1_temp)\
            + t1_outf[ngr - 1]*numpy.exp(1.j*(tir[ngr - 1] - tib[y])*D1)

    t1_outi = numpy.zeros(t1bari.shape, dtype=complex)
    dt = numpy.zeros((ngi))
    for y in range(ngi):
        for i in range(y):
            dt[i] = tii[i] - tii[y]
        gtemp = numpy.exp(dt[:, None, None]*D1[None, :, :])
        t1_temp = gtemp*t1bari
        t1_outi[y] = einsum('x,xai->ai', Gi[y], t1_temp) + t1_outb[ngr - 1]*numpy.exp(D1*(1.j*tir[0] - tii[y]))

    return t1_outf, t1_outb, t1_outi


def int_tbar2_keldysh(ngr, ngi, t2barf, t2barb, t2bari, tir, tii, D2, Gr, Gi):
    """Integrate t1bar with exponential factor."""
    t2_outf = numpy.zeros(t2barf.shape, dtype=complex)
    dt = numpy.zeros((ngr))
    for y in range(ngr):
        for i in range(ngr):
            dt[i] = tir[i] - tir[y]
        gtemp = numpy.exp(1.j*dt[:, None, None, None, None]*D2[None, :, :, :, :])
        t2_temp = gtemp*t2barf
        t2_outf[y] = 1.j*einsum('x,xabij->abij', Gr[y], t2_temp)

    t2_outb = numpy.zeros(t2barb.shape, dtype=complex)
    dt = numpy.zeros((ngr))
    tib = tir.copy()
    for i in range(ngr):
        tib[ngr - i - 1] = tir[i]
    for y in range(ngr):
        for i in range(ngr):
            dt[i] = tib[i] - tib[y]
        gtemp = numpy.exp(1.j*dt[:, None, None, None, None]*D2[None, :, :, :, :])
        t2_temp = gtemp*t2barb
        t2_outb[y] = -1.j*einsum('x,xabij->abij', Gr[y], t2_temp)\
            + t2_outf[ngr - 1]*numpy.exp(1.j*(tir[ngr - 1] - tib[y])*D2)

    t2_outi = numpy.zeros(t2bari.shape, dtype=complex)
    dt = numpy.zeros((ngi))
    for y in range(ngi):
        for i in range(y):
            dt[i] = tii[i] - tii[y]
        gtemp = numpy.exp(dt[:, None, None, None, None]*D2[None, :, :, :, :])
        t2_temp = gtemp*t2bari
        t2_outi[y] = einsum('x,xabij->abij', Gi[y], t2_temp) + t2_outb[ngr - 1]*numpy.exp(D2*(1.j*tir[0] - tii[y]))

    return t2_outf, t2_outb, t2_outi


def int_L1_keldysh(ngr, ngi, L1f, L1b, L1i, tir, tii, D1, gr, gi, Gr, Gi):
    """Return L1bar."""
    L1i_out = numpy.zeros(L1i.shape, dtype=complex)
    for s in range(ngi):
        dt = numpy.zeros((ngi))
        for y in range(s, ngi):
            dt[y] = tii[s] - tii[y]
        gtemp = numpy.exp(dt[:, None, None]*D1[None, :, :])
        L1temp = einsum('yai,yia->yia', gtemp, L1i)
        L1i_out[s] = einsum('y,yia,y->ia', gi, L1temp, Gi[:, s])/gi[s]

    L1b_out = numpy.zeros(L1b.shape, dtype=complex)
    tib = tir.copy()
    for i in range(ngr):
        tib[ngr - i - 1] = tir[i]
    for s in range(ngr):
        dt = numpy.zeros((ngr))
        for y in range(s, ngr):
            dt[y] = tib[s] - tib[y]
        gtemp = numpy.exp(1.j*dt[:, None, None]*D1[None, :, :])
        L1temp = einsum('yai,yia->yia', gtemp, L1b)
        add = numpy.exp(D1[None, :, :]*(1.j*tib[s] - tii[:, None, None]))/gr[s]
        L1b_out[s] = -1.j*einsum('y,yia,y->ia', gr, L1temp, Gr[:, s])/gr[s]
        L1b_out[s] += numpy.einsum('y,yia,yai->ia', gi, L1i, add)*Gr[ngr - 1, s]

    L1f_out = numpy.zeros(L1b.shape, dtype=complex)
    for s in range(ngr):
        dt = numpy.zeros((ngr))
        for y in range(s, ngr):
            dt[y] = tir[s] - tir[y]
        gtemp = numpy.exp(1.j*dt[:, None, None]*D1[None, :, :])
        L1temp = einsum('yai,yia->yia', gtemp, L1f)
        fac = numpy.exp(1.j*D1[None, :, :]*(tir[s] - tib[:, None, None]))/gr[s]
        add = numpy.exp(D1[None, :, :]*(1.j*tir[s] - tii[:, None, None]))/gr[s]
        L1f_out[s] = 1.j*einsum('y,yia,y->ia', gr, L1temp, Gr[:, s])/gr[s]
        L1f_out[s] -= 1.j*numpy.einsum('y,yia,yai->ia', gr, L1b, fac)*Gr[ngr - 1, s]
        L1f_out[s] += numpy.einsum('y,yia,yai->ia', gi, L1i, add)*Gr[ngr - 1, s]

    return L1f_out, L1b_out, L1i_out


def int_L2_keldysh(ngr, ngi, L2f, L2b, L2i, tir, tii, D2, gr, gi, Gr, Gi):
    """Return L2bar."""
    L2i_out = numpy.zeros(L2i.shape, dtype=complex)
    for s in range(ngi):
        dt = numpy.zeros((ngi))
        for y in range(s, ngi):
            dt[y] = tii[s] - tii[y]
        gtemp = numpy.exp(dt[:, None, None, None, None]*D2[None, :, :, :, :])
        L2temp = einsum('yabij,yijab->yijab', gtemp, L2i)
        L2i_out[s] = einsum('y,yijab,y->ijab', gi, L2temp, Gi[:, s])/gi[s]

    L2b_out = numpy.zeros(L2b.shape, dtype=complex)
    tib = tir.copy()
    for i in range(ngr):
        tib[ngr - i - 1] = tir[i]
    for s in range(ngr):
        dt = numpy.zeros((ngr))
        for y in range(s, ngr):
            dt[y] = tib[s] - tib[y]
        gtemp = numpy.exp(1.j*dt[:, None, None, None, None]*D2[None, :, :, :, :])
        L2temp = einsum('yabij,yijab->yijab', gtemp, L2b)
        add = numpy.exp(D2[None, :, :, :, :]*(1.j*tib[s] - tii[:, None, None, None, None]))/gr[s]
        L2b_out[s] = -1.j*einsum('y,yijab,y->ijab', gr, L2temp, Gr[:, s])/gr[s]
        L2b_out[s] += numpy.einsum('y,yijab,yabij->ijab', gi, L2i, add)*Gr[ngr - 1, s]

    L2f_out = numpy.zeros(L2b.shape, dtype=complex)
    for s in range(ngr):
        dt = numpy.zeros((ngr))
        for y in range(s, ngr):
            dt[y] = tir[s] - tir[y]
        gtemp = numpy.exp(1.j*dt[:, None, None, None, None]*D2[None, :, :, :, :])
        L2temp = einsum('yabij,yijab->yijab', gtemp, L2f)
        fac = numpy.exp(1.j*D2[None, :, :, :, :]*(tir[s] - tib[:, None, None, None, None]))/gr[s]
        add = numpy.exp(D2[None, :, :, :, :]*(1.j*tir[s] - tii[:, None, None, None, None]))/gr[s]
        L2f_out[s] = 1.j*einsum('y,yijab,y->ijab', gr, L2temp, Gr[:, s])/gr[s]
        L2f_out[s] -= 1.j*numpy.einsum('y,yijab,yabij->ijab', gr, L2b, fac)*Gr[ngr - 1, s]
        L2f_out[s] += numpy.einsum('y,yijab,yabij->ijab', gi, L2i, add)*Gr[ngr - 1, s]

    return L2f_out, L2b_out, L2i_out
