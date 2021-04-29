import numpy
from cqcpy import utils

def scaled_energy(x):
    """Return the square the reduced k-vector (x)."""
    return x[0]*x[0] + x[1]*x[1] + x[2]*x[2]

def computek(x, L):
    """Return the actual k-vector given a reduced vector (x)."""
    ccc = 2.0*numpy.pi/L
    return numpy.array((ccc*x[0],ccc*x[1],ccc*x[2]))

def computeE(k):
    """Return the energy of a given k-vector."""
    return (k[0]*k[0] + k[1]*k[1] + k[2]*k[2]) / 2.0

class ueg_basis(object):
    """UEG plane-wave basis set.

    Attributes:
        L (float): Box length.
        cutoff (float): Energy cutoff.
        basis (array): array of reduced k-vectors.
        Es (array): array of energies.
        ks (array): array of k-vectors.
    """
    def __init__(self, L, cutoff, norb=None):
        self.L = L
        self.cutoff = cutoff
        kmax = numpy.sqrt(2*cutoff)
        imaxf = L*kmax / (2*numpy.pi)
        imax = int(numpy.ceil(imaxf) + 0.1)
        self.basis = []
        for i in range(-imax,imax):
            for j in range(-imax,imax):
                for l in range(-imax,imax):
                    k2 = 4.0*numpy.pi*numpy.pi / (L*L) \
                            * (i*i + j*j + l*l)
                    if k2 < 2.0*cutoff:
                        self.basis.append((i,j,l))

        self.basis.sort(key=scaled_energy)
        if norb is not None:
            self.basis = self.basis[:norb]
        self.Es = []
        self.ks = []
        for x in self.basis:
            k = computek(x,L)
            self.ks.append(k)
            self.Es.append(computeE(k))

    def get_nbsf(self):
        return len(self.basis)

    def build_r_ke_matrix(self):
        diag = numpy.asarray(self.Es)
        return numpy.diag(diag)

    def build_u_ke_matrix(self):
        diag = numpy.asarray(self.Es)
        T = numpy.diag(diag)
        return T,T

    def build_g_ke_matrix(self):
        diag = numpy.asarray(self.Es)
        T = numpy.diag(diag)
        return utils.block_diag(T,T)

    def r_build_diag(self):
        return numpy.asarray(self.Es)

    def u_build_diag(self):
        return numpy.asarray(self.Es),numpy.asarray(self.Es)

    def g_build_diag(self):
        return numpy.hstack((
            numpy.asarray(self.Es),numpy.asarray(self.Es)))

    def build_r2e_matrix(self):
        n = self.get_nbsf()
        V = numpy.zeros((n,n,n,n))
        L = self.L
        aaa = 4.0*numpy.pi / (L*L*L)
        for p in range(n):
            kp = self.ks[p]
            for q in range(n):
                kq = self.ks[q]
                for r in range(n):
                    kr = self.ks[r]
                    for s in range(n):
                        ks = self.ks[s]
                        peqr = (p == r)
                        qeqs = (q == s)
                        s1 = True#(p == r)
                        s2 = True#(q == s)
                        if (not peqr) and (not qeqs) and s1 and s2:
                            if numpy.linalg.norm(kp + kq - kr - ks) < 1e-12:
                                V[p,q,r,s] = aaa / (scaled_energy(kp - kr))
        return V

    def build_u2e_matrix(self, anti=True):
        V = self.build_r2e_matrix()
        if anti:
            Va = V - numpy.transpose(V,(0,1,3,2))
        return Va,Va,V


    def build_g2e_matrix(self, anti=True):
        nbsf = self.get_nbsf()
        n = 2*nbsf
        V = numpy.zeros((n,n,n,n))
        L = self.L
        aaa = 4.0*numpy.pi / (L*L*L)
        for p in range(n):
            kp = self.ks[p % nbsf]
            sp = p//nbsf
            for q in range(n):
                kq = self.ks[q % nbsf]
                sq = q//nbsf
                for r in range(n):
                    kr = self.ks[r % nbsf]
                    sr = r//nbsf
                    for s in range(n):
                        ss = s//nbsf
                        ks = self.ks[s % nbsf]
                        peqr = (p == r)
                        qeqs = (q == s)
                        s1 = (sp == sr)
                        s2 = (sq == ss)
                        if (not peqr) and (not qeqs) and s1 and s2:
                            if numpy.linalg.norm(kp + kq - kr - ks) < 1e-12:
                                V[p,q,r,s] = aaa / (scaled_energy(kp - kr))

        if anti:
            return V - numpy.transpose(V,(0,1,3,2))
        else:
            return V
