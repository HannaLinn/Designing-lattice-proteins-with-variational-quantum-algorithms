from DesignProteinClass import *
from utils import *
from dataset import *
from hamiltonian import *
from QAOA import *

from scipy import optimize
import qiskit
import qiskit_aer
from qiskit import transpile

import time

current_timestamp = time.time()
import os

current_file_directory = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path

target_folder = Path("./references/")

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

verbose = True

p_min = 1
p_max = 15
maxiter = 10000
donation_of_params = True

# interpolate, noise, XY, fc, start in all

versions = [#[True, True, True, True, False],
            #[True, True, True, False, True],
            [True, True, True, True, True],
            [True, True, False, False, True],
            # no noise
            [True, False, True, True, False],
            [True, False, True, False, True],
            [True, False, True, True, True],
            [True, False, False, False, True],
            ]


# To generate a dataset of all design problems
n = 8
colors = plt.cm.coolwarm(np.linspace(0, 1, n))
G_list = set_of_design_problems(n=n, all=False)
v_num = 0

for v in versions:
    v_num += 1
    interpolate = v[0]
    noise = v[1]
    XY = v[2]
    fully_connected = v[3]
    start_in_all = v[4]

    if XY:
        lamda = 0.0
        now_testing = "XY"
        if fully_connected:
            now_testing = now_testing + "_fc"
        else:
            now_testing = now_testing + "_not_fc"

        if start_in_all:
            now_testing = now_testing + "_startinall"
    else:
        lamda = 2.0
        now_testing = "X"
        start_in_all = False

    if noise:
        now_testing = now_testing + "_noise"

    now_testing = now_testing + "_"
    if donation_of_params:
        now_testing = now_testing + "_donation_of_params_"

    print(now_testing, v_num)
    save_dir = current_file_directory + "/results/" + now_testing + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cost_vector = np.zeros((n, p_max))
    success_vector = np.zeros((n, p_max))
    depth_vector_after = np.zeros((n, p_max))
    depth_vector_before = np.zeros((n, p_max))
    depth_vector_after_1qb = np.zeros((n, p_max))
    depth_vector_after_2qb = np.zeros((n, p_max))
    depth_vector_before_1qb = np.zeros((n, p_max))
    depth_vector_before_2qb = np.zeros((n, p_max))
    num_qubits_vector = np.zeros(n)

    for i in range(n):
        instance = G_list[i]
        G = instance[0]
        NH = instance[1][0]
        print("\n i: ", i, " of ", n)
        print(G, NH)

        # Create an instance of the problem
        design_instance = DesignProteinInstance(
            G, NH=NH, lamda=lamda
        )  # lambda should be set dep. on XY

        design_instance.calc_solution_sets()

        num_qubits = design_instance.num_bits
        num_qubits_vector[i] = num_qubits
        name = "num_qb_" + str(num_qubits) + "_" + "NH_" + str(NH)

        # TRANSFORM INTO HAMILTONIAN
        # Make into Hamiltonian
        J_dict, h_dict = QUBO_to_Ising(num_qubits, design_instance.Q)
        H_cost = get_cost_hamiltonian(num_qubits, J_dict, h_dict)

        eigenvalues = get_eigenvalues(H_cost)
        ground_energy, ground_states_i = get_ground_states_i(
            design_instance.feasible_set, H_cost
        )

        # BUILD A CIRCUIT
        qubits = list(G.nodes())

        for p in range(p_min, p_max + 1):
            print("\np ", p, ". Out of: ", p_max)
            # print('bp \n', best_params_bp)
            
            # bounds = [(-np.pi * 4, np.pi * 4)] * p + [(-np.pi * 4, np.pi * 4)] * p
            if donation_of_params and p == p_min and i > 0:
                temp_G_item = G_list[i-1]
                temp_G = temp_G_item[0]  # Graph
                temp_qb = len(temp_G.nodes())
                temp_nh = temp_G_item[1][0]  # Number of H-bonds
                name_of_smaller = "num_qb_" + str(temp_qb) + "_" + "NH_" + str(temp_nh)
                init_params = np.genfromtxt(
                    save_dir + "best_params_p_" + str(p_min) + name_of_smaller + ".out", delimiter=","
                )
                print("donating from: ", name_of_smaller)
                print("donated params: ", init_params)

            elif interpolate and p != p_min:
                init_params = best_params

                print("given_init_params: \n", init_params)
                init_params = interpolate_params(init_params, only_last=True)
                print("next params: \n", init_params)
        
            else:
                bounds = [(0, np.pi * 2)] * p + [(0, np.pi * 2)] * p
                init_params = [(bound[0] - bound[-1]) / 2 for bound in bounds]

            info_dict = {
                "XY": XY,
                "maxiter": maxiter,
                "G": G,
                "NH": NH,
                "fully connected": fully_connected,
                "current timestamp": current_timestamp,
                "now_testing": now_testing,
                "init_params": init_params,
                "bound": bounds,
            }

            if XY:
                if start_in_all:
                    qaoa_circuit = get_qaoa_circuit(
                        qubits,
                        init_circuit_XY,
                        J_dict,
                        h_dict,
                        mixer_circuit_XY,
                        p,
                        fully_connected=fully_connected,
                        feasible_solution_set=design_instance.feasible_set,
                    )
                else:
                    qaoa_circuit = get_qaoa_circuit(
                        qubits,
                        init_circuit_XY,
                        J_dict,
                        h_dict,
                        mixer_circuit_XY,
                        p,
                        fully_connected=fully_connected,
                        feasible_solution=design_instance.feasible_set[0],
                    )
            else:
                qaoa_circuit = get_qaoa_circuit(
                    qubits,
                    init_circuit_X,
                    J_dict,
                    h_dict,
                    mixer_circuit_X,
                    p,
                )

            depth_vector_before[i, p - 1] = qaoa_circuit.depth()
            depth_vector_before_1qb[i, p - 1] = count_gates(qaoa_circuit)[1]
            depth_vector_before_2qb[i, p - 1] = count_gates(qaoa_circuit)[2]
            np.savetxt(
                save_dir + "depth_vector_before_p_" + str(p) + name + ".out",
                depth_vector_before,
                delimiter=",",
            )
            np.savetxt(
                save_dir + "depth_vector_before_1qb" + str(p) + name + ".out",
                depth_vector_before_1qb,
                delimiter=",",
            )
            np.savetxt(
                save_dir + "depth_vector_before_2qb" + str(p) + name + ".out",
                depth_vector_before_2qb,
                delimiter=",",
            )

            if not noise:  # noiseless
                device = FakeTorino()
                backend = qiskit_aer.AerSimulator(method="statevector",
                                            )
                qaoa_circuit = qiskit.transpile(qaoa_circuit, backend,
                    #coupling_map = device.coupling_map,
                    optimization_level=3, seed_transpiler=1)
                #backend.set_options(device='GPU')

            else:
                #print(qaoa_circuit)
                device = FakeTorino()
                #device = FakeTorino()
                noise_model = NoiseModel.from_backend(device)
                backend = qiskit_aer.AerSimulator(method="statevector", noise_model=noise_model,
                                            )
                qaoa_circuit = qiskit.transpile(qaoa_circuit, backend,
                    #coupling_map = device.coupling_map,
                    optimization_level=3, seed_transpiler=1)
                #backend.set_options(device='GPU')


            args = (
                qaoa_circuit,
                backend,
                eigenvalues,
            )

            # TRAIN
            result = optimize.minimize(
                average_cost,
                x0=init_params,
                args=args,
                method="COBYLA",
                bounds=bounds,
                options={"disp": True, "maxiter": maxiter},
            )
            best_params = result.x

            cost_vector[i, p - 1] = average_cost(best_params, *args)
            success_vector[i, p - 1] = success_prob(
                best_params, qaoa_circuit, backend, ground_states_i
            )

            save_info(info_dict, save_dir + "optimising_coeffs.txt")

            print("Average cost of best parameters: ", average_cost(best_params, *args))
            print(
                "Success probability of best parameters: ",
                success_prob(best_params, qaoa_circuit, backend, ground_states_i),
            )

            np.savetxt(
                save_dir + "best_params_p_" + str(p) + name + ".out",
                best_params,
                delimiter=",",
            )
            np.savetxt(
                save_dir + "success_vector_p_" + str(p) + name + ".out",
                success_vector,
                delimiter=",",
            )
            np.savetxt(
                save_dir + "cost_vector_p_" + str(p) + name + ".out",
                cost_vector,
                delimiter=",",
            )
            '''
            depth_vector_after[i, p - 1] = qaoa_circuit.depth()
            depth_vector_after_1qb[i, p - 1] = count_gates(qaoa_circuit)[1]
            depth_vector_after_2qb[i, p - 1] = count_gates(qaoa_circuit)[2]
            np.savetxt(
                save_dir + "depth_vector_after_p_" + str(p) + name + ".out",
                depth_vector_after,
                delimiter=",",
            )
            np.savetxt(
                save_dir + "depth_vector_after_1qb" + str(p) + name + ".out",
                depth_vector_after_1qb,
                delimiter=",",
            )
            np.savetxt(
                save_dir + "depth_vector_after_2qb" + str(p) + name + ".out",
                depth_vector_after_2qb,
                delimiter=",",
            )
            '''

        # PLOTS

        plt.figure(11)
        plt.title(
            now_testing
            + " Success probability for increasing p, inter: "
            + str(interpolate)
        )
        plt.plot(
            [str(s) for s in np.arange(p_min, p_max + 1)],
            success_vector[i, :],
            label=name,
        )
        plt.legend()
        plt.xlabel("p")
        plt.ylabel("Success probability")
        plt.ylim((0, 1))
        plt.savefig(save_dir + "Increasing_p" + ".pdf")

        plt.figure(10)
        plt.plot([str(int(s)) for s in num_qubits_vector[:i]], success_vector[:i, -1])
        plt.title(now_testing + " Dataset, p = " + str(p))
        plt.xlabel("Number of qubits")
        plt.ylabel("Success probability")
        plt.ylim((0, 1))
        plt.savefig(
            save_dir + "success_over_qubits_p_" + str(p) + ".pdf", bbox_inches="tight"
        )
        print("\n")
    plt.figure(10)
    plt.plot([str(int(s)) for s in num_qubits_vector], success_vector[:, -1])
    plt.title(now_testing + " Dataset, p = " + str(p))
    plt.xlabel("Number of qubits")
    plt.ylabel("Success probability")
    plt.ylim((0, 1))
    plt.savefig(
        save_dir + "success_over_qubits_p_" + str(p) + ".pdf", bbox_inches="tight"
    )
    print("\n")
