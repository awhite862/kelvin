import logging
import sys as csys
from kelvin.ueg_system import UEGSystem
from kelvin.ccsd import ccsd

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    stream=csys.stdout)

T = 0.0
na = 7
nb = 7
N = na + nb
L = 3.88513
mu = 2.0
ueg = UEGSystem(T, L, 9.4, mu=mu, norb=33, orbtype='u', madelung='const')
logging.info('density: {}'.format(ueg.den))
logging.info('r_s: {}'.format(ueg.rs))
cc = ccsd(ueg, T=T, mu=mu, iprint=1)
E = cc.run()
logging.info("{}".format(E))
ueg = UEGSystem(T, L, 9.4, mu=mu, norb=33, orbtype='u', madelung='orb')
logging.info('density: {}'.format(ueg.den))
logging.info('r_s: {}'.format(ueg.rs))
cc = ccsd(ueg, T=T, mu=mu, iprint=1)
E = cc.run()
logging.info("{}".format(E))
