import numpy as np

from cell_simulator.dynamics import HarmonicOscillatorContraction
from cell_simulator.integrators import RK4


def test_rk4_harmonic_oscillator_matches_sine():
    """RK4 on the harmonic oscillator must reproduce c(t) = sin(omega t + phi).

    Catches a broken integrator or a sign error in drift().
    """
    omega, phase = 2 * np.pi / 10.0, 0.8
    dyn = HarmonicOscillatorContraction(omega, phase)
    times, states = RK4().integrate(dyn, 0.0, 30.0, 3000, dyn.initial_state())
    np.testing.assert_allclose(states[:, 0], np.sin(omega * times + phase), atol=1e-6)
