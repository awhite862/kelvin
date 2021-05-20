import logging
import sys as csys
import numpy
import h5py
from kelvin.solid_field_system import solid_field_system
from kelvin.neq_ccsd import neq_ccsd
from pyscf.pbc.gto import Cell
from pyscf.pbc.tools.lattice import get_ase_atom
from pyscf.pbc import scf

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    stream=csys.stdout)

formula = 'si'
bas = 'gth-szv'

ase_atom = get_ase_atom(formula)

cell = Cell()
if formula.lower() == 'si':
    ke_cutoff = 40
elif formula.lower() == 'c':
    ke_cutoff = 40
elif formula.lower() == 'ge':
    ke_cutoff = 40
else:
    raise RuntimeError(
        'No recommended kinetic energy cutoff for formula ', formula)
cell.from_ase(ase_atom).build(
    unit='B', ke_cutoff=ke_cutoff, basis=bas,
    precision=1e-8, verbose=9, pseudo='gth-pade')
mf = scf.RHF(cell, exxdiv=None)
ehf = mf.kernel()
logging.info("HF energy (per unit cell) = %.17g" % ehf)

T = 0.2
mu = 0.5527
A0 = 40*0.0168803
t0 = 15.0
sigma = 2.0
omega = 0.97529
tmax = 30.0
ng = 200
ngi = 20
deltat = tmax / ng

ti = numpy.asarray([deltat/2 + float(j)*deltat for j in range(ng)])
sys = solid_field_system(T, mf, ti, A0, t0, sigma, omega, mu=mu)
cc = neq_ccsd(
    sys, T, mu=mu, tmax=tmax, econv=1e-8, max_iter=1000,
    damp=0.6, ngr=ng, ngi=ngi, iprint=1)
E, Ecc = cc.run()
cc._neq_ccsd_lambda()
cc._neq_1rdm()
p = cc.compute_1rdm()

hf = h5py.File('den20_040.h5', 'w')
hf.create_dataset("density", data=p)
hf.close()
