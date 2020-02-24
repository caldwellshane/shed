import numpy as np
import qutip as qu

from shed.hamiltonians import transmon_hamiltonian


def simulate_pi_pulse(
        anharm: float,
        rabi_freq: float,
        time_points: np.ndarray,
        drive_phase: float = 0
) -> qu.solver.Result:
    """
    Assume transmon driven on resonance.

    Assuming a resonant drive allows us to avoid having time dependence in the Hamiltonian.

    :param anharm: Anharmonicity of the transmon, in linear frequency. Should be negative.
    :param rabi_freq: Frequency of Rabi oscillations. Proportional to drive strength, and in same
    units as ``anharm``.
    :param time_points: Times at which to capture the wavefunction.
    :drive_phase: Phase of the drive tone, in radians.
    :return: A ``Result`` containing the sampled wavefunctions in the ``states`` attribute.
    """
    dim = 10
    a = qu.destroy(dim)
    at = a.dag()

    # Build Hamiltonian
    H0 = transmon_hamiltonian(0, anharm, dim)
    H1 = 0.5 * rabi_freq * (np.cos(drive_phase) * (a + at) - 1j * np.sin(drive_phase) * (a - at))
    H = H0 + H1

    # Evolve state per Schrodinger equation
    return qu.mesolve(2 * np.pi * H, qu.basis(dim, 0), time_points)
