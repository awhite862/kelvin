class neq_system(object):
    """Base class representing a physical system (deprecated)."""
    def __init__(self):
        pass

    def reversible(self):
        """Is there forward-backward symmetry in the dynamics?."""
        err = "Base class function verify()"
        raise Exception(err)

    def verify(self,T,mu):
        """Verify that the system is consistent with the given T,mu."""
        err = "Base class function verify()"
        raise Exception(err)

    def const_energy(self):
        """Return the constant energy shift if there is one."""
        err = "Base class function const_energy()"
        raise Exception(err)

    def get_mp1(self):
        """Return the 1st order energy."""
        err = "Base class function get_mp1()"
        raise Exception(err)

    def g_energies_tot(self):
        """Return the general spin orbital energies in occ and virt blocks."""
        err = "Base class function g_energies()"
        raise Exception(err)

    def g_fock_tot(self,direc='f'):
        """Return the general 1-electron operator (Fock matrix including diagonal)."""
        err = "Base class function g_fock_tot()"
        raise Exception(err)

    def g_int_tot(self):
        """Return the general orbital symmetric 2-electron operator."""
        err = "Base class function g_int_tot()"
        raise Exception(err)

class NeqSystem(object):
    """Base class representing a physical system."""
    def __init__(self):
        pass

    def verify(self,T,mu):
        """Verify that the system is consistent with the given T,mu."""
        err = "Base class function verify()"
        raise Exception(err)

    def const_energy(self):
        """Return the constant energy shift if there is one."""
        err = "Base class function const_energy()"
        raise Exception(err)

    def get_mp1(self):
        """Return the 1st order energy."""
        err = "Base class function get_mp1()"
        raise Exception(err)

    def g_energies_tot(self):
        """Return the general spin orbital energies in occ and virt blocks."""
        err = "Base class function g_energies()"
        raise Exception(err)

    def g_fock_tot(self,t=0.0):
        """Return the general 1-electron operator (Fock matrix including diagonal)."""
        err = "Base class function g_fock_tot()"
        raise Exception(err)

    def g_int_tot(self):
        """Return the general orbital symmetric 2-electron operator."""
        err = "Base class function g_int_tot()"
        raise Exception(err)
