import qutip as qu


def transmon_hamiltonian(freq: float, anharm: float, dim: int=3) -> qu.Qobj:
    """
    A transmon truncated to a number of dimensions.

    This is represented as a Duffing oscillator, where the Kerr term is assumed small.
    """
    a = qu.destroy(dim)
    at = a.dag()
    return freq * at * a + 0.5 * anharm * at * at * a * a

