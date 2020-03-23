import numpy as np
import qutip as qu

from shed.hamiltonians import qubit_hamiltonian, transmon_hamiltonian


def evolve_xy_driven_qubit(
    time_points: np.ndarray,
    qubit_freq: float,
    rabi_freq: float,
    drive_freq: float,
    drive_phase: float = 0
) -> qu.solver.Result:
    """
    Time-evolve a qubit under an XY drive.

    :param time_points: Times at which to capture the wavefunction.
    :param qubit_freq: Qubit frequency. Set to zero to be in the rotating frame.
    :param rabi_freq: Frequency of Rabi oscillations.
    :param drive_freq: Frequency of qubit drive.
    :param drive_phase: Phase of qubit drive, relative to qubit phase which is zero.
    :return: A ``Result`` containing the sampled wavefunctions in the ``states`` attribute.
    """
    H0 = qubit_hamiltonian(qubit_freq)
    H1 = rabi_freq * qu.sigmay()
    H = [
        2 * np.pi * H0,
        [
            2 * np.pi * H1,
            f"cos({2 * np.pi * drive_freq} * t + {2 * np.pi * drive_phase})"
        ]
    ]

    # Evolve state per Schrodinger equation
    return qu.mesolve(H, qu.basis(2, 0), time_points)


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
