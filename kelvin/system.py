class System(object):
    """Base class representing a physical system."""
    def __init__(self):
        pass

    def verify(self, T, mu):
        """Verify that the system is consistent with the given T,mu."""
        err = "Base class function verify()"
        raise Exception(err)

    def has_g(self):
        """Implementation of generalized orbital quantities"""
        err = "Base class function has_g()"
        raise Exception(err)

    def has_u(self):
        """Implementation of unrestricted orbital quantities"""
        err = "Base class function has_u()"
        raise Exception(err)

    def has_r(self):
        """Implementation of restricted orbital quantities"""
        err = "Base class function has_r()"
        raise Exception(err)

    def const_energy(self):
        """Return the constant energy shift if there is one."""
        err = "Base class function const_energy()"
        raise Exception(err)

    def get_mp1(self):
        """Return the 1st order energy."""
        err = "Base class function get_mp1()"
        raise Exception(err)

    def g_d_mp1(self, dvec):
        """Return the derivative of the MP1 energy with respect to
        occupation numbers."""
        err = "Base class function g_d_mp1()"
        raise Exception(err)

    def u_d_mp1(self, dveca, dvecb):
        """Return the derivative of the MP1 energy with respect to
        occupation numbers."""
        err = "Base class function u_d_mp1()"
        raise Exception(err)

    def r_energies(self):
        """Return the restricted orbital energies in occ and virt blocks."""
        err = "Base class function r_energies()"
        raise Exception(err)

    def u_energies(self):
        """Return the general spin orbital energies in occ and virt blocks."""
        err = "Base class function u_energies()"
        raise Exception(err)

    def g_energies(self):
        """Return the general spin orbital energies in occ and virt blocks."""
        err = "Base class function g_energies()"
        raise Exception(err)

    def r_energies_tot(self):
        """Return all the R orbital energies."""
        err = "Base class function r_energies_tot()"
        raise Exception(err)

    def u_energies_tot(self):
        """Return all the U orbital energies."""
        err = "Base class function u_energies_tot()"
        raise Exception(err)

    def g_energies_tot(self):
        """Return all the U/G orbital energies."""
        err = "Base class function g_energies_tot()"
        raise Exception(err)

    def r_fock(self):
        """Return the restricted Fock operator (including diagonal)
        in block form."""
        err = "Base class function r_fock()"
        raise Exception(err)

    def u_fock(self):
        """Return the unrestricted Fock operators (including diagonal)
        in block form."""
        err = "Base class function u_fock()"
        raise Exception(err)

    def g_fock(self):
        """Return the general Fock operator (including diagonal)
        in block form."""
        err = "Base class function g_fock()"
        raise Exception(err)

    def r_fock_tot(self):
        """Return the general 1-electron operator
        (Fock matrix including diagonal)."""
        err = "Base class function r_fock_tot()"
        raise Exception(err)

    def u_fock_tot(self):
        """Return the unrestricted 1-electron operators
        (Fock matrix including diagonal)."""
        err = "Base class function u_fock_tot()"
        raise Exception(err)

    def g_fock_tot(self):
        """Return the general 1-electron operator
        (Fock matrix including diagonal)."""
        err = "Base class function g_fock_tot()"
        raise Exception(err)

    def g_fock_d_tot(self, dvec):
        """Return the n-derivative of the general 1-electron operator ."""
        err = "Base class function g_fock_d_tot()"
        raise Exception(err)

    def u_fock_d_tot(self, dveca, dvecb):
        """Return the n-derivative of the general 1-electron operator ."""
        err = "Base class function u_fock_d_tot()"
        raise Exception(err)

    def r_hcore(self):
        """Return the restricted 1-electron operator
        (Core Hamiltonian matrix)."""
        err = "Base class function r_hcore()"
        raise Exception(err)

    def g_hcore(self):
        """Return the general orbital 1-electron operator
        (Core Hamiltonian matrix)."""
        err = "Base class function g_hcore()"
        raise Exception(err)

    def g_aint(self, code=0):
        """Return general orbital blocks of the anti-symmetrized 2-electron
        interaction."""
        err = "Base class function g_aint()"
        raise Exception(err)

    def u_aint(self):
        """Return unrestricted blocks of the anti-symmetrized 2-electron
        interaction."""
        err = "Base class function u_aint()"
        raise Exception(err)

    def u_aint_tot(self):
        """Return unrestricted blocks of the anti-symmetrized 2-electron
        interaction."""
        err = "Base class function u_aint_tot()"
        raise Exception(err)

    def g_aint_tot(self):
        """Return the general orbital anti-symmetrized 2-electron operator."""
        err = "Base class function g_aint_tot()"
        raise Exception(err)

    def r_int_tot(self):
        """Return the restricted orbital 2-electron operator."""
        err = "Base class function r_int_tot()"
        raise Exception(err)

    def g_int_tot(self):
        """Return the general orbital symmetric 2-electron operator."""
        err = "Base class function g_int_tot()"
        raise Exception(err)
