from DesignProteinClass import *
from utils import *
from dataset import *
from hamiltonian import *
from QAOA import *
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from scipy import optimize
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pickle



# FUNCTIONS --------------------------------------------------------------------

def save_constants_to_file(file_path: str, constants_dict: dict) -> None:
    with open(file_path, "w") as f:
        for key, value in constants_dict.items():
            f.write(f"{key}: {value}\n")


def calculate_batch_energy(
    conf_bitstring_list: list[str],
    energy_function: callable,
    energy_args: tuple,
) -> list[float]:
    """
    Calculate the energy of a batch of bitstrings.

    Args:
        conf_bitstring_list (list[str]): List of conformation bitstrings.
        energy_function (callable): The energy function.
        energy_args (tuple): The arguments of the energy function.

    Returns:
        list[float]: List of energies of the batch of bitstrings.
    """
    return [
        energy_function(bitstring, **energy_args) for bitstring in conf_bitstring_list
    ]

def energy_func(state, hamiltonian_matrix, nh, lamda):
    E_HP = 0
    E_cnstr = 0
    arr = np.array([int(i) for i in state])[::-1]
    E_HP = arr @ hamiltonian_matrix @ arr
    E_cnstr += lamda * (sum(arr) - nh) ** 2

    return E_HP + E_cnstr

def bit_flip_correction(string):
    bit_flip_strings=[]
   
    for i in range(len(string)):
        s=list(string)
        s[i]=str(1-int(string[i]))
        string_new=''.join(s)
        bit_flip_strings.append(string_new)
    return bit_flip_strings

def batch_energy(dict_of_strings, hamiltonian_matrix, nh, lamda, bit_flip=True):
    E=0
    bit_flip_dict={'1':0}
    
    for string, count in dict_of_strings.items():
        e=energy_func(string,hamiltonian_matrix,nh,lamda)
        if bit_flip:
            bit_flip_list=bit_flip_correction(string)
            for bit_flip_string in bit_flip_list:
                e_bit_flip=energy_func(string,hamiltonian_matrix,nh,lamda)
                if e_bit_flip<e:
                     e=e_bit_flip
                     dict_of_strings[bit_flip_string] = dict_of_strings.pop(string)
                     
        Energy_list.append(e)
        Solution_list.append(string)
        E+=e*count
        
    tot_counts = sum(list(dict_of_strings.values()))
    
    
    return E / tot_counts


def cost_func(params, ansatz, sampler, hamiltonian_matrix, nh, lamda, save_file_at, bit_flip):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [params])
    job = sampler.run(pubs=[pub])
    primitive_result = job.result()
    pub_result = primitive_result[0].data
    counts = pub_result.meas.get_counts()
    energy = batch_energy(counts, hamiltonian_matrix, nh, lamda, bit_flip)

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")
    with open(save_file_at + 'cost_history_dict.pkl', 'wb') as f:
        pickle.dump(cost_history_dict, f)

    return energy


# CONSTANTS --------------------------------------------------------------------

verbose = True
noisy_simulation = False
simulate = False
lamda = 1.1
num_proteins = 10
maxiter = 100
average_iter = 5
real_amp = False
num_layers = 1
default_shots = 100_000
bit_flip = True


# RUNS -------------------------------------------------------------------------
color = plt.cm.coolwarm(np.linspace(0.1, 0.9, num_proteins))

current_file_directory = os.path.dirname(os.path.abspath(__file__))

if simulate:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer.primitives import SamplerV2 as Sampler
    import qiskit_aer
    device = FakeTorino()
    backend = qiskit_aer.AerSimulator(
        method="statevector",
    )

    if noisy_simulation:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(device)
        pass_manager = generate_preset_pass_manager(3, backend)
        sampler = Sampler(options=dict(backend_options=dict(noise_model=noise_model)))
    else:
        sampler = Sampler(default_shots=default_shots)
else:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Session
    from qiskit_ibm_runtime import QiskitRuntimeService

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="-",
    token="-",
    )
    # backend = service.backend("ibm_torino")
    backend = service.least_busy(simulator=False, operational=True)
    print("Backend: ", backend)
    
    pass_manager = generate_preset_pass_manager(3, backend)

now_testing = (
    backend.name + "_same_num_layers_" + str(num_layers) + "_" + str(default_shots) + "shots" + "_real" + str(real_amp) + "_noisy_simulation_" + str(noisy_simulation) + "_bit_flip_" + str(bit_flip) + "/"
)
save_file_at = current_file_directory + "/results/" + now_testing

# Ensure the directory exists
os.makedirs(os.path.dirname(save_file_at), exist_ok=True)

tick = time.time()

# To generate a dataset of all design problems
G_list = set_of_design_problems(n=4, all=True)
data_set = G_list[:num_proteins]

constants_to_save = {
    "data_set": data_set,
    "simulate": simulate,
    "noisy_simulation": noisy_simulation,
    "maxiter": maxiter,
    "average_iter": average_iter,
    "num_layers": num_layers,
    "lamda": lamda,
    "default_shots": default_shots,
    "bit_flip": bit_flip
}

# Save constants to file
save_constants_to_file(save_file_at + "constants_dict.txt", constants_to_save)

hitrate_dict = {}
counter = 0
for i, G_item in enumerate(data_set):
    np.random.seed(10 + counter)
    counter += 1
    G = G_item[0]  # Graph
    nh = G_item[1][0]  # Number of H-bonds
    ground_energy = G_item[2][0]
    ground_state_string = G_item[3][0]
    name = str(nh) + "_" + str(i)

    # Create an instance of the problem
    design_instance = DesignProteinInstance(G, NH=nh, lamda=lamda)
    print("Q:", design_instance.Q)

    design_instance.calc_solution_sets()
    # print('solution set: ', design_instance.solution_set)
    # print('feasible set: ', design_instance.feasible_set)

    num_qubits = design_instance.num_bits
    print("Number of qubits: ", num_qubits)

    if real_amp:
        ansatz = RealAmplitudes(num_qubits, reps=num_layers).decompose()
    else:
        ansatz = EfficientSU2(num_qubits, reps=num_layers).decompose()
    ansatz.measure_all()

    hamiltonian_matrix = np.zeros((len(G.nodes()), len(G.nodes())))
    for u, v in list(G.edges()):
        hamiltonian_matrix[u][v] -= 1 / 2
        hamiltonian_matrix[v][u] -= 1 / 2

    num_params = ansatz.num_parameters

    if not simulate or noisy_simulation:
        ansatz = pass_manager.run(ansatz)

    # run many runs with random init_params, save the best one
    cost_trajectory_list = []
    energy_lowest_array = np.zeros(average_iter)
    hitrate_array = np.zeros(average_iter)
    for average_i in range(average_iter): # benchmarking
        Energy_list = []
        Solution_list = []
        # reset the cost_history_dict
        cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        init_params = 2 * np.pi * np.random.random(num_params)
        if simulate:
            res = optimize.minimize(
                cost_func,
                init_params,
                args=(ansatz, sampler, hamiltonian_matrix, nh, lamda, save_file_at, bit_flip),
                options={"maxiter": maxiter, "disp": True},
                method="cobyla",
            )
            params = cost_history_dict["prev_vector"]
            pub = (ansatz, [params])
            job = sampler.run(pubs=[pub])
            primitive_result = job.result()
            pub_result = primitive_result[0].data
            counts = pub_result.meas.get_counts()
            # print(counts, params)
            print("Final cost: ", cost_history_dict["cost_history"][-1])
        else:
            with Session(backend=backend) as session:
                sampler = Sampler(mode=session)
                sampler.options.default_shots = default_shots
                res = optimize.minimize(
                    cost_func,
                    init_params,
                    args=(ansatz, sampler, hamiltonian_matrix, nh, lamda, save_file_at, bit_flip),
                    options={"maxiter": maxiter, "disp": True},
                    method="cobyla",
                )
                params = cost_history_dict["prev_vector"]
                pub = (ansatz, [params])
                job = sampler.run(pubs=[pub])
                primitive_result = job.result()
                pub_result = primitive_result[0].data
                counts = pub_result.meas.get_counts()
                # print(counts, params)
                print("Final cost: ", cost_history_dict["cost_history"][-1])


        # Save the cost trajectory
        np.save(
            save_file_at + name + "_cost_trajectory_av" + str(average_i) + ".npy",
            cost_history_dict["cost_history"],
        )
        cost_trajectory_list.append(cost_history_dict["cost_history"])

        lowest_energy = min(Energy_list)
        best_bitstring = Solution_list[Energy_list.index(lowest_energy)]
        hit = 0
        tot_counts = 0
        for string, count in counts.items():
            if energy_func(string, hamiltonian_matrix, nh, lamda) == lowest_energy:
                hit = count
            tot_counts += count
        hit /= tot_counts
        print(lowest_energy, best_bitstring, hit)
        energy_lowest_array[average_i] = lowest_energy

        hitrate_array[average_i] = hit
        # TODO: spara bara de 100 bästa
        np.save(save_file_at + name + "Solutions_list" + str(average_i) + ".npy", Solution_list)
        np.save(save_file_at + name + "params" + str(average_i) + ".npy", params)

    # Find the lowest energy
    lowest_energy_i = np.argmin(energy_lowest_array)
    print("energy_lowest_array ", energy_lowest_array)
    print("lowest_energy_i ", lowest_energy_i)
    lowest_energy = energy_lowest_array[lowest_energy_i]
    print("lowest_energy ", lowest_energy)
    lowest_cost_trajectory = cost_trajectory_list[lowest_energy_i]

    # Save the hitrate
    hitrate = sum(hitrate_array) / average_iter
    print(hitrate, ground_energy, lowest_energy, best_bitstring)
    hitrate_dict[num_qubits] = (hitrate, ground_energy, lowest_energy, best_bitstring)
    np.save(save_file_at + "hitrate_dict.npy", hitrate_dict)

    # Plot the cost trajectory
    plot_cost(
        lowest_cost_trajectory,
        name=save_file_at + name + "_cost_trajectory",
        pdf=True,
        new_fig=False,
        ground_energy=None,
        label=nh,
        color=color[i],
    )

    print()


# Plot hitrates over number of aa
# plt.figure()
# for i, p in enumerate(hitrate_dict.values()):
#     plt.plot(p[0], p[1], "o", color=color[i], label=list(hitrate_dict.keys())[i])
# xticks = np.unique([p[0] for p in hitrate_dict.values()])
# plt.xticks(xticks, [str(a) for a in xticks])
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.xlabel("Number of amino acids")
# plt.ylabel("Difference in energy to ground energy")
# plt.savefig(save_file_at + "diffs_plot.pdf", bbox_inches="tight")

tack = time.time()
duration = tack - tick

print(f" >> The whole calculation took {duration/60:.2f} minutes")
print("Files are here: ", save_file_at)
