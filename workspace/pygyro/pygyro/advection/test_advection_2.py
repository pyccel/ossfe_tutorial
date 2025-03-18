from mpi4py import MPI
import pytest
from scipy.integrate import trapezoid
import numpy as np

from ..initialisation.setups import setupCylindricalGrid
from ..model.layout import Layout
from ..initialisation.initialiser_funcs import f_eq as fEq
from .advection import FluxSurfaceAdvection, PoloidalAdvection, VParallelAdvection, ParallelGradient
from .. import splines as spl
from ..initialisation.constants import Constants


def gauss(x):
    return np.exp(-x**2/4)


@pytest.mark.serial
@pytest.mark.parametrize("fact,dt", [(10, 1), (10, 0.1), (5, 1)])
def test_fluxSurfaceAdvection(fact, dt):
    npts = [30, 20]
    eta_vals = [np.linspace(0, 1, 4), np.linspace(0, 2*np.pi, npts[0], endpoint=False),
                np.linspace(0, 20, npts[1], endpoint=False), np.linspace(0, 1, 4)]

    N = 10

    f_vals = np.ndarray(npts)

    domain = [[0, 2*np.pi], [0, 20]]
    nkts = [n+1 for n in npts]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, 3, True) for b in breaks]
    bsplines = [spl.BSplines(k, 3, True, True) for k in knots]
    eta_grids = [bspl.greville for bspl in bsplines]

    c = 2

    eta_vals[1] = eta_grids[0]
    eta_vals[2] = eta_grids[1]
    eta_vals[3][0] = c

    layout = Layout('flux', [1], [0, 3, 1, 2], eta_vals, [0])

    constants = Constants()

    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    f_vals[:, :] = np.sin(eta_vals[2]*np.pi/fact)
    f_end = np.sin((eta_vals[2]-c*dt*N)*np.pi/fact)

    for n in range(N):
        fluxAdv.step(f_vals, 0)

    assert np.max(np.abs(f_vals-f_end)) < 1e-4


@pytest.mark.serial
@pytest.mark.parametrize("nptZ,dt,err", [(32, 1.0, 1.5e-2), (64, 0.5, 3e-4), (128, 0.25, 1e-5)])
def test_fluxSurfaceAdvectionAligned(nptZ, dt, err):
    npts = [nptZ, nptZ]

    constants = Constants()
    constants.iotaVal = 0.8
    constants.n = -11

    eta_vals = [np.linspace(0, 1, 4), np.linspace(0, 2*np.pi, npts[0], endpoint=False),
                np.linspace(0, 2*np.pi*constants.R0, npts[1], endpoint=False), np.linspace(0, 1, 4)]

    N = 10

    f_vals = np.ndarray(npts)

    domain = [[0, 2*np.pi], [0, 2*np.pi*constants.R0]]
    nkts = [n+1 for n in npts]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, 3, True) for b in breaks]
    bsplines = [spl.BSplines(k, 3, True, True) for k in knots]
    eta_grids = [bspl.greville for bspl in bsplines]

    c = 2

    eta_vals[1] = eta_grids[0]
    eta_vals[2] = eta_grids[1]
    eta_vals[3][0] = c

    layout = Layout('flux', [1], [0, 3, 1, 2], eta_vals, [0])

    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    m, n = (5, -4)
    theta = eta_grids[0]
    phi = eta_grids[1]*2*np.pi/domain[1][1]
    f_vals[:, :] = np.sin(m*theta[:, None] + n*phi[None, :])

    # ~ f_vals[:,:] = np.sin(eta_vals[2]*np.pi/fact)
    f_end = f_vals.copy()

    for n in range(N):
        fluxAdv.step(f_vals, 0)
    # ~ print(np.max(np.abs(f_vals-f_end)))
    assert np.max(np.abs(f_vals-f_end)) < err
