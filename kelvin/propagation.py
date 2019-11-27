import numpy

def trk1(h, t1, t2, RHS):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    t1 += k1s
    t2 += k1d

def trk2(h, t1, t2, RHS):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    k2s,k2d = RHS(t1 + k1s, t2 + k1d)
    k2s *= h
    k2d *= h
    t1 += 0.5*(k1s + k2s)
    t2 += 0.5*(k1d + k2d)

def trk4(h, t1, t2, RHS):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    k2s,k2d = RHS(t1 + 0.5*k1s, t2 + 0.5*k1d)
    k2s *= h
    k2d *= h
    k3s,k3d = RHS(t1 + 0.5*k2s, t2 + 0.5*k2d)
    k3s *= h
    k3d *= h
    k4s,k4d = RHS(t1 + k3s, t2 + k3d)
    k4s *= h
    k4d *= h
    t1 += 1.0/6.0*(k1s + 2.0*k2s + 2.0*k3s + k4s)
    t2 += 1.0/6.0*(k1d + 2.0*k2d + 2.0*k3d + k4d)

def tab2(h, t1, t2, k2s, k2d, RHS):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    if k2s is None and k2d is None:
        t1 += k1s
        t2 += k1d
    else:
        t1 += (1.5*k1s - 0.5*k2s)
        t2 += (1.5*k1d - 0.5*k2d)
    return k1s,k1d

def tbe_step(h, t1, t2, k1s, k1d, mi, alpha, thresh, RHS, iprint):
    dsold = k1s
    ddold = k1d
    if iprint > 1:
        print("Time step {}:".format(i))
    converged = False
    for k in range(mi):
        ds,dd = RHS(t1 + dsold, t2 + ddold)
        ds *= h
        dd *= h
        error = numpy.linalg.norm(ds - dsold)/(numpy.linalg.norm(dsold) + 0.01)
        error += numpy.linalg.norm(dd - ddold)/(numpy.linalg.norm(ddold) + 0.01)
        if iprint > 1:
            print(' %2d  %.4E' % (k+1,error))
        dsold = (1.0 - alpha)*ds + alpha*dsold
        ddold = (1.0 - alpha)*dd + alpha*ddold
        if error < thresh: 
            converged = True
            break
    if not converged: raise Exception("BE: Failed to compute implicit step")
    return dsold, ddold

def tbe(h, t1, t2, mi, alpha, thresh, RHS, iprint):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    ds, dd = tbe_step(h, t1, t2, k1s, k1d, mi, alpha, thresh, RHS, iprint)
    t1 += ds
    t2 += dd

def tcn(h, t1, t2, mi, alpha, thresh, RHS, iprint):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    ds, dd = tbe_step(h, t1, t2, k1s, k1d, mi, alpha, thresh, RHS, iprint)
    t1 += 0.5*(ds + k1s)
    t2 += 0.5*(dd + k1d)

def tam2(h, t1, t2, mi, alpha, thresh, RHS, iprint):
    k1s,k1d = RHS(t1,t2)
    k1s *= h
    k1d *= h
    dsold = k1s
    ddold = k1d
    if iprint > 1:
        print("Time step {}:".format(i))
    converged = False
    for k in range(mi):
        ds,dd = RHS(t1 + dsold, t2 + ddold)
        ds *= h
        dd *= h
        ds = 0.5*(ds + k1s)
        dd = 0.5*(dd + k1d)
        error = numpy.linalg.norm(ds - dsold)/(numpy.linalg.norm(dsold) + 0.01)
        error += numpy.linalg.norm(dd - ddold)/(numpy.linalg.norm(ddold) + 0.01)
        if iprint > 1:
            print(' %2d  %.4E' % (k+1,error))
        dsold = (1.0 - alpha)*ds + alpha*dsold
        ddold = (1.0 - alpha)*dd + alpha*ddold
        if error < thresh: 
            converged = True
            break
    if not converged: raise Exception("AM2: Failed to compute implicit step")
    t1 += dsold
    t2 += ddold
