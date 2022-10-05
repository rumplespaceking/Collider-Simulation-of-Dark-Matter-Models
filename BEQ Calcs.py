import matplotlib.pyplot as plot
import numpy as np
from scipy.special import kn, k1
from scipy.integrate import odeint, solve_ivp, BDF, Radau
from scipy.integrate import quad
from numpy import pi
from math import sqrt
#from numba import jit
#from mpmath import odefun


mphi, mpl, mh, lmb = 300, 1.22*(10**(19)), 125, 0.01

#@jit(nopython=True)
def sig(s,mphi,lmb,mh,x):
    sph = (lmb**2)*(1 + (3*(mh**2)/(s - mh**2)))*(k1((x/mphi)*sqrt(s)))*(sqrt(s - 4*(mh**2)))*(sqrt(s - 4*(mphi**2)))*((sqrt(s))**(-1))
    return sph

#@jit(nopython=True)
def sigvt(x):
    if mphi > mh:
        sigvt=(x/(256*(mphi**5)*(kn(2,x)*kn(2,x))*np.pi))*(quad(sig, 4*(mphi**(2)),10**(8),args=(mphi,lmb,mh,x))[0])
    else:
        sigvt=(x/(256*(mphi**5)*(kn(2,x)*kn(2,x))))*(quad(sig, 4*(mh**(2)),10**(8),args=(mphi,lmb,mh,x))[0])
    return sigvt

c1 = (45/(4*pi**4))
c2 = -((2*(pi**2))/(45*1.67))*mpl*sigvt(25)

#@jit(nopython=True)
def yeq(x):
    eq = c1*(x**2)*kn(2,x)
    return eq

#@jit(nopython=True)
def beq(x,y):
    dydx = c2*(mphi/(x**2))*(y*y - yeq(x)*yeq(x))
    return dydx


xspan = np.array([0.1, 100])
y0 = np.array([yeq(0.1), yeq(1), yeq(10)])
solver = 'BDF'
n = 0.01
y = (2.74385*(10**8))*(mphi)*((solve_ivp(beq, xspan, y0, method=solver, max_step = n, dense_output=True).y))
xt = (solve_ivp(beq, xspan, y0, method=solver,  max_step = n, dense_output=True).t)
y1 = (2.74385*(10**8))*(mphi)*yeq(xt)
#(2.74385*(10**8))*
plot.loglog(xt,y[0],label='Y')
plot.loglog(xt,y1,label='$Y_{EQ}$')
plot.xlabel('$x=\dfrac{m_{\phi}}{T}$', fontsize=16)
plot.ylabel('$\Omega_{DM} h^2$', fontsize=16)
plot.xlim([0.1,1e2])
plot.ylim([10**(-1),10**(12)])
plot.legend()
plot.show()

relden = 2.344e-5*(y[0][-1])
print("The calculated relic density for this model is:", relden)