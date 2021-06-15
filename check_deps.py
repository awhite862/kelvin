import sys

no_pyscf = False
try:
    import pyscf
except ImportError:
    no_pyscf = True
    print("Failed to import 'pyscf'")  

no_lattice = False
try:
    import lattice
except ImportError:
    no_lattice = True
    print("Failed to import 'lattice' library")  

no_cqcpy = False
try:
    import cqcpy
except ImportError:
    no_cqcpy = True
    print("Failed to import 'cqcpy' library")  

if no_lattice or no_cqcpy or no_pyscf:
    sys.exit(1)
