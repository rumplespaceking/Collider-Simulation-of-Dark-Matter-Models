

# from math import *
# from numpy import *
import matplotlib.pyplot as plot
import numpy as np
from scipy.special import kn
from scipy.integrate import odeint
from scipy.integrate import quad
from numpy import pi
from math import sqrt, log
import scipy.interpolate as spi

data0 = np.loadtxt("grho.dat")
data1 = np.loadtxt("gs.dat")
x0 = data0[:, 0]
y0 = data0[:, 1]
x1 = data1[:, 0]
y1 = data1[:, 1]
gst = spi.interp1d(x1, y1, fill_value="extrapolate")
grh = spi.interp1d(x0, y0, fill_value="extrapolate")

# mass of DM particle
mphi = 200
gdm = 1.0
mpl = 1.22 * (10 ** (19))
init = .01
mh = 125.1
lphih = 0.01
vev = 246.0


def sigmaphiphihh(s, mphi, lphih, mh, vev, x):
    sphiphihh = (sqrt(s)) * (s - 4 * (mphi ** 2)) * (kn(1, (x / mphi) * (sqrt(s)))) * (
                (1 / (16 * pi * (-4 * (mh ** 4) + (-2 * (mh ** 2) + s) ** 2))) * \
                (1 / 2) * s * (sqrt(-(((4 * (mh ** 2) - s) * (-4 * (mphi ** 2) + s)) / (s ** 2))) * \
                               ((lphih ** 2) + (9 * (lphih ** 2) * (mh ** 4)) / (((mh ** 2) - s) ** 2) - \
                                (6 * (lphih ** 2) * (mh ** 2)) / ((mh ** 2) - s) - (8 * (lphih ** 4) * (vev ** 4)) /
                                (-4 * (mh ** 4) + 4 * (mh ** 2) * s + (s ** 2) * (-1 - ((4 * (mh ** 2) - s) * \
                                                                                        (-4 * (mphi ** 2) + s)) / (
                                                                                              s ** 2)))) + 2 * (
                                           lphih ** 2) * (vev ** 2) * (lphih - \
                                                                       lphih * vev * ((3 * (mh ** 2)) / (
                                    ((mh ** 2) - s) * vev) - (lphih * vev) / (2 * (mh ** 2) - \
                                                                              s))) * log(
                    (-2 * (mh ** 2) + s) - s * (sqrt(-(((4 * (mh ** 2) - s) * (-4 * (mphi ** 2) + \
                                                                               s)) / (s ** 2)))) / (
                                -2 * (mh ** 2) + s + s * (
                            sqrt(-(((4 * (mh ** 2) - s) * (-4 * (mphi ** 2) + s)) / (s ** 2))))))))
    return sphiphihh


def sigmavtphiphihh(x):
    if mphi > mh:
        svtphiphihh = (x / (8 * (mphi ** 4) * mphi * (kn(2, x) * kn(2, x)))) * (
        quad(sigmaphiphihh, 4 * (mphi ** (2)), 10 ** (8), args=(mphi, lphih, mh, vev, x))[0])
    else:
        svtphiphihh = (x / (8 * (mphi ** 4) * mphi * (kn(2, x) * kn(2, x)))) * (
        quad(sigmaphiphihh, 4 * (mh ** (2)), 10 ** (8), args=(mphi, lphih, mh, vev, x))[0])
    return svtphiphihh


def eq(x):
    equiv = (45 / (4 * pi ** 4)) * (gdm / (gst(mphi / x))) * (x ** 2) * kn(2, x)
    return equiv


def beq(y, x):
    dydx = -((2 * (pi ** 2)) / (45 * 1.67)) * mpl * ((gst(mphi / x)) / (sqrt(grh(mphi / x)))) * (mphi / (x ** 2)) * (
                y * y - eq(x) * eq(x)) * sigmavtphiphihh(25)
    return dydx


#x0 = np.linspace(init, 100, 10000)
#y0 = eq(init)
x0 = np.array([0.1, 100])
y0 = np.array([eq(0.01), eq(0.1)])
# solve ODE
#y = (mphi) * (2.74385 * (10 ** 8)) * odeint(beq, y0, x0, hmax=0.01)
solver = 'BDF'
n = 0.01
y = (2.74385*(10**8))*(mphi)*((solve_ivp(beq, x0, y0, method=solver, max_step = n, dense_output=True).y))
y1 = (mphi) * (2.74385 * (10 ** 8)) * eq(x0)
xt = (solve_ivp(beq, x0, y0, method=solver, max_step = n, dense_output=True).t)
# plot results
plot.loglog(xt, y, label='DM')
plot.loglog(xt, y1, label='Eq')
plot.xlabel('$x=\dfrac{m_{DM}}{T}$', fontsize=16)
plot.ylabel('$\Omega_{DM} h^2$', fontsize=16)
plot.xlim([0.1, 100])
plot.ylim([10 ** (-1), 10 ** (10)])
plot.legend()
plot.show()

