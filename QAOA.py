from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
import numpy as np
from qiskit_aer import backends

# CIRCUITS


def get_qaoa_circuit(
    qubits: list,
    init_circuit: QuantumCircuit,
    J_dict: dict[tuple[int, int], float],
    h_dict: dict[int, float],
    mixer_hamiltonian: QuantumCircuit,
    p: int,
    **kwargs,
) -> QuantumCircuit:
    """
    Input:
        - init_circuit = function of circuit
        - J_dict (dict[tuple[int, int], Number]): Two-body interaction energies J_{ij}.
        - h_dict (dict[int, Number], optional): One-body local fields h_i.
        - mixer_hamiltonian = function of circuit
        - p = int Number of layers in the qaoa algorithm
        - **kwargs:     fully_connected = boolean if the XY mixer will be fully connected or sequencial
                        hamming_weight = integer of the hamming weight for the feasible solution
    """
    circuit = init_circuit(qubits, kwargs)
    gammas = [Parameter(f"gamma_{i}") for i in range(p)]
    betas = [Parameter(f"beta_{i}") for i in range(p)]

    for layer in range(p):
        circuit = circuit.compose(cost_circuit(J_dict, h_dict, qubits, gammas[layer]))
        circuit = circuit.compose(mixer_hamiltonian(qubits, betas[layer], kwargs))
    return circuit


def cost_circuit(
    J_dict: dict[tuple[int, int], float],
    h_dict: dict[int, float],
    qubits: list,
    gamma: Parameter,
) -> QuantumCircuit:
    circuit = QuantumCircuit(len(qubits))
    for (i, j), J_ij in J_dict.items():
        if J_ij:
            circuit.rzz(
                2 * J_ij * gamma, i, j
            )  # *2 because the gate is defined with angle/2 https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZZGate
    for i, h_i in h_dict.items():
        if h_i:
            circuit.rz(2 * h_i * gamma, i)
    return circuit


# Mixers


def mixer_circuit_X(qubits: list, beta: Parameter, kwargs) -> QuantumCircuit:
    circuit = QuantumCircuit(len(qubits))

    for q in qubits:
        circuit.rx(2 * beta, q)
    return circuit


def mixer_circuit_XY(qubits: list, beta: Parameter, kwargs) -> QuantumCircuit:
    """
    is qubits always range(num_qubits) ?
    """
    circuit = QuantumCircuit(len(qubits))
    # beta = Parameter("beta")
    if kwargs["fully_connected"]:  # all qubits connected to each other
        for q1 in qubits:
            for q2 in qubits:
                if q1 >= q2:
                    continue
                circuit.rxx(2 * beta, q1, q2)
                circuit.ryy(2 * beta, q1, q2)

    else:
        for q in qubits:
            if q != qubits[-1]:
                circuit.rxx(2 * beta, q, q + 1)
                circuit.ryy(2 * beta, q, q + 1)
            else:
                circuit.rxx(2 * beta, q, 0)
                circuit.ryy(2 * beta, q, 0)
    return circuit


def dicke_state(n: int, k: int, qc: QuantumCircuit) -> QuantumCircuit:
    for i in range(0, k):
        qc.x(i)

    for j in range(0, n - 1):
        qc.cx(j + 1, j)
        qc.cry(2 * np.arccos(np.sqrt(1 / (n - j))), j, j + 1)
        qc.cx(j + 1, j)
        last = j + 1
        if last != n - 1:
            num = 2
            for i in range(last + 1, j + k + 1):
                if i < n:
                    qc.cx(i, j)
                    qc.mcry(2 * np.arccos(np.sqrt(num / (n - j))), [j, last], i)
                    qc.cx(i, j)
                    num += 1
                    last += 1
    # print(qc)
    return qc.decompose()


def init_circuit_X(qubits: list, kwargs) -> QuantumCircuit:
    """
    Starts in a superposition of all solutions.
    Input:
        - qubits = a list of all qubits
    """
    init_circuit = QuantumCircuit(len(qubits))
    for q in qubits:
        init_circuit.h(q)
    return init_circuit


def init_circuit_XY(qubits: list, kwargs) -> QuantumCircuit:
    # TODO make a superposition of all states with the given hamming
    # weight. Use https://arxiv.org/pdf/1904.07358.pdf
    if "hamming_weight" in kwargs.keys():
        hamming_weight = kwargs["hamming_weight"]
        feasible_solution = [1] * hamming_weight + [0] * (len(qubits) - hamming_weight)
    elif "feasible_solution" in kwargs.keys():
        feasible_solution = kwargs["feasible_solution"]

        init_circuit = QuantumCircuit(len(qubits))
        for qubit in qubits:
            if feasible_solution[qubit] == 1:
                init_circuit.x(qubit)

    elif "feasible_solution_set" in kwargs.keys():  # Dicke state
        init_circuit = QuantumCircuit(len(qubits))
        hamming_weight = sum(kwargs["feasible_solution_set"][0])
        init_circuit = dicke_state(len(qubits), hamming_weight, init_circuit)

    else:
        print("Give feasible solution of hamming weight.")

    return init_circuit


# COST FUNCTIONS


def set_parameters(
    params: list | np.ndarray,
    qaoa_circuit: QuantumCircuit,
) -> QuantumCircuit:
    p = int(len(qaoa_circuit.parameters) / 2)
    gammas = [f"gamma_{i}" for i in range(p)]
    betas = [f"beta_{i}" for i in range(p)]
    params_dict = dict(zip(gammas + betas, params))
    return qaoa_circuit.assign_parameters(params_dict)


def average_cost(
    params: list | np.ndarray,
    qaoa_circuit: QuantumCircuit,
    backend: backends.aer_simulator.AerSimulator,
    eigenvalues: np.ndarray,
) -> float:
    """
    Input:
        params = list of the parameters for the circuit, [all gammas, all betas]
    """
    qaoa_circuit = set_parameters(params, qaoa_circuit)

    qaoa_circuit.save_statevector()
    result = backend.run(qaoa_circuit).result().get_statevector().data
    return np.real(np.vdot(result, np.multiply(eigenvalues, result)))


def expectation_value_hamiltonian(
    params: list | np.ndarray,
    qaoa_circuit: QuantumCircuit,
    backend: backends.aer_simulator.AerSimulator,
    hamiltonian: SparsePauliOp,
) -> float:
    """
    Does the same thing as average_cost when called with H_cost, but can be called with another hamiltonian.

    Input:
        params = list of the parameters for the circuit, [all gammas, all betas]

    """
    qaoa_circuit = set_parameters(params, qaoa_circuit)
    #qaoa_circuit = qaoa_circuit.measure_all()

    qaoa_circuit = qaoa_circuit.save_statevector()
    result = backend.run(qaoa_circuit).result().get_statevector()
    exp = Statevector(result).expectation_value(hamiltonian)
    return np.real(exp)


def get_probabilities(
    params: list | np.ndarray,
    qaoa_circuit: QuantumCircuit,
    backend: backends.aer_simulator.AerSimulator,
) -> np.ndarray:
    """
    Input:
        params = list of the parameters for the circuit, [all gammas, all betas]
    """
    qaoa_circuit = set_parameters(params, qaoa_circuit)

    qaoa_circuit.save_statevector()  # Save the simulator state as a statevector
    result = backend.run(qaoa_circuit).result().get_statevector()
    probs = Statevector(result).probabilities()
    return probs


def success_prob(
    params: list | np.ndarray,
    qaoa_circuit: QuantumCircuit,
    backend: backends.aer_simulator.AerSimulator,
    ground_states_i: np.ndarray,
) -> float:
    """
    Input:
        params = list of the parameters for the circuit, [all gammas, all betas]
    """
    probs = get_probabilities(params, qaoa_circuit, backend)
    return np.sum(probs[ground_states_i])
