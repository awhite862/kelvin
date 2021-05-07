import numpy


def rk1(h, var, RHS):
    out = RHS(var)
    return [h*x for x in out]


def rk1_gen(t0, y0, dt, func):
    k1 = func(t0, y0)
    return [dt*k for k in k1]


def rk2(h, var, RHS):
    k1 = RHS[0](var)
    k1 = [h*x for x in k1]
    k2 = RHS[1]([x + y for x,y in zip(k1,var)])
    return [0.5*(x + h*y) for x,y in zip(k1,k2)]


def rk2_gen(t0, y0, dt, func):
    k1 = func(t0, y0)
    k2 = func(t0 + 0.5*dt, [y + 0.5*dt*x for y,x in zip(y0, k1)])
    return [0.5*dt*(x + y) for x,y in zip(k1,k2)]


def rk4(h, var, RHS):
    k1 = RHS[0](var)
    k1 = [h*x for x in k1]

    var2 = [x + 0.5*y for x,y in zip(var,k1)]
    k2 = RHS[1](var2)
    k2 = [h*x for x in k2]

    var3 = [x + 0.5*y for x,y in zip(var,k2)]
    k3 = RHS[2](var3)
    k3 = [h*x for x in k3]

    var4 = [x + y for x,y in zip(var,k3)]
    k4 = RHS[3](var4)
    k4 = [h*x for x in k4]

    return [1.0/6.0*(x + 2.0*y + 2.0*z + a) for x,y,z,a in zip(k1,k2,k3,k4)]


def rk4_gen(t0, y0, dt, func):
    k1 = func(t0, y0)
    k2 = func(t0 + 0.5*dt, [a + 0.5*dt*b for a,b in zip(y0, k1)])
    k3 = func(t0 + 0.5*dt, [a + 0.5*dt*b for a,b in zip(y0, k2)])
    k4 = func(t0 + dt, [a + dt*b for a,b in zip(y0, k3)])
    return [dt*(a + 2*b + 2*c + d)/6.0 for a,b,c,d in zip(k1,k2,k3,k4)]


def ab2(h, var, k2, RHS):
    k1 = RHS(var)
    k1 = [h*x for x in k1]
    if k2[0] is None:
        return k1,k1
    else:
        d = [1.5*x - 0.5*y for x,y in zip(k1,k2)]
    return d,k1


def be_step(h, var, k1, mi, alpha, thresh, RHS, iprint):
    dold = k1
    converged = False
    for k in range(mi):
        d = RHS([x + y for x,y in zip(var,dold)])
        d = [h*x for x in d]
        error = sum([numpy.linalg.norm(x - y)/(numpy.linalg.norm(x) + 0.01) for x,y in zip(d,dold)])
        if iprint > 1:
            print(' %2d  %.4E' % (k+1,error))
        dold = [(1.0 - alpha)*x + alpha*y for x,y in zip(d,dold)]
        if error < thresh:
            converged = True
            break
    if not converged:
        raise Exception("BE: Failed to compute implicit step")
    return dold


def be(h, var, mi, alpha, thresh, RHS, iprint):
    k1 = RHS(var)
    k1 = [h*x for x in k1]
    d = be_step(h, var, k1, mi, alpha, thresh, RHS, iprint)
    return d


def cn(h, var, mi, alpha, thresh, RHS, iprint):
    k1 = RHS[0](var)
    k1 = [h*x for x in k1]
    d = be_step(h, var, k1, mi, alpha, thresh, RHS[1], iprint)
    return [0.5*(x + y) for x,y in zip(d,k1)]


def am2(h, var, mi, alpha, thresh, RHS, iprint):
    k1 = RHS(var)
    k1 = [h*x for x in k1]
    dold = k1
    converged = False
    for k in range(mi):
        d = RHS([x + y for x,y in zip(var,dold)])
        d = [0.5*(h*x + y) for x,y in zip(d,k1)]
        error = sum([numpy.linalg.norm(x - y)/(numpy.linalg.norm(x) + 0.01) for x,y in zip(d,dold)])
        if iprint > 1:
            print(' %2d  %.4E' % (k+1,error))
        dold = [(1.0 - alpha)*x + alpha*y for x,y in zip(d,dold)]
        if error < thresh:
            converged = True
            break
    if not converged:
        raise Exception("AM2: Failed to compute implicit step")
    return dold
