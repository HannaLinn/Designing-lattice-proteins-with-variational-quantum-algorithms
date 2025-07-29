import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import combinations
from qiskit.quantum_info import Pauli, SparsePauliOp
from typing import Tuple


def QUBO_to_Ising(
    num_qubits: int, Q: dict[tuple[int, int], float]
) -> Tuple[dict[tuple[int, int], float], dict[int, float]]:
    """
    Transform a QUBO to an Ising formulation of the problem.

    Input:
            - Q = QUBO [Dictionary with keys : (qubit i, qubit j) as integers, value : (energy as float)]
                    Q needs to top triangular!
            - num_qubits (int): Number of qubits in circuit.

    Output:
            - J_dict (dict[tuple[int, int], float]): Two-body interaction energies J_{ij}.
            - h_dict (dict[int, float]): One-body local fields h_i.
    """
    J_dict = defaultdict(float)  # type: dict[tuple[int, int], float]
    h_dict = defaultdict(float)  # type: dict[int, float]

    # Symmetric Q
    Q_copy = defaultdict(float)  # type: dict[tuple[int, int], float]
    for j in range(num_qubits):
        for k in range(num_qubits):
            if j > k:
                Q_copy[(j, k)] = Q[(k, j)]
            else:
                Q_copy[(j, k)] = Q[(j, k)]
    Q_copy = dict(Q_copy)

    # One-body energies, h_dict:
    # \sum^N_{i=1}
    for i in range(num_qubits):
        h_dict[(i)] = -Q_copy[(i, i)] / 2  # - \frac{O_is_i}{2}

    # Two-body energies, J_dict:
    # \sum^{N-1}{i=1}\sum^N{j=i+1}
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            J_dict[(i, j)] += Q_copy[(i, j)] / 4  # \frac{s_is_j T_{ij}}{4}
            h_dict[(i)] -= Q_copy[(i, j)] / 4  # - \frac{s_i T_{ij}}{4}
            h_dict[(j)] -= Q_copy[(i, j)] / 4  # - \frac{s_j T_{ij}}{4}
    return dict(J_dict), dict(h_dict)


def get_cost_hamiltonian(
    num_qubits: int, J_dict: dict[tuple[int, int], float], h_dict: dict[int, float]
) -> SparsePauliOp:
    """
    Creates the full Hamiltonian.
    """
    H_cost_O = get_h_hamiltonian(num_qubits, h_dict)
    H_cost_T = get_J_hamiltonian(num_qubits, J_dict)
    return H_cost_O + H_cost_T


def get_h_hamiltonian(num_qubits: int, h_dict: dict[int, float]) -> SparsePauliOp:
    """
    Creates the one-body terms of the cost Hamiltonian.
    """
    pauli_list = []
    for i in range(num_qubits):
        z = np.zeros(num_qubits, dtype=bool)
        z[i] = True
        pauli_list.append(Pauli((z, np.zeros(num_qubits, dtype=bool))))
    return SparsePauliOp(pauli_list, coeffs=list(h_dict.values()))


def get_J_hamiltonian(
    num_qubits: int, J_dict: dict[tuple[int, int], float]
) -> SparsePauliOp:
    """
    Creates the two-body terms of the cost Hamiltonian.
    """
    pauli_list = []
    coeffs = []
    for j in range(num_qubits):
        for k in range(j + 1, num_qubits):
            coeffs.append(J_dict[j, k])
            z = np.zeros(num_qubits, dtype=bool)
            z[j] = True
            z[k] = True
            pauli_list.append(Pauli((z, np.zeros(num_qubits, dtype=bool))))
    return SparsePauliOp(pauli_list, coeffs=coeffs)
