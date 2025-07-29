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

plot = True

generate = False

start_gamma = 0.0
stop_gamma = 2.0 * np.pi
num_points_gamma = 20

start_beta = 0.0
stop_beta = 2.0 * np.pi
num_points_beta = 20

p = 1

fig_num = 1

# To generate a dataset of all design problems
n = 8
G_list = set_of_design_problems(n=n, all=False)
v_num = 0

# interpolate, noise, XY, fc, start in all

versions = [#[True, True, True, True, False],
            #[True, True, True, False, True],
            #[True, True, True, True, True],
            #[True, True, False, False, True],
            # no noise
            [True, False, True, False, False],
            #[True, False, True, True, False],
            #[True, False, True, False, True],
            #[True, False, True, True, True],
            #[True, False, False, False, True],
            ]

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

    print(now_testing, v_num)
    save_dir = current_file_directory + "/results/" + "landscapes/" + now_testing + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cost_vector = np.zeros((n, p))
    success_vector = np.zeros((n, p))
    depth_vector_after = np.zeros((n, p))
    depth_vector_before = np.zeros((n, p))
    depth_vector_after_1qb = np.zeros((n, p))
    depth_vector_after_2qb = np.zeros((n, p))
    depth_vector_before_1qb = np.zeros((n, p))
    depth_vector_before_2qb = np.zeros((n, p))
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
        print(name)

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
        if generate:
            info_dict = {
                "XY": XY,
                "G": G,
                "NH": NH,
                "fully connected": fully_connected,
                "current timestamp": current_timestamp,
                "now_testing": now_testing,
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

            from qiskit_ibm_runtime import EstimatorV2 as Estimator
            from qiskit_ibm_runtime import Session
            with Session(backend=backend) as session:
                estimator = Estimator(mode=session)
            
                # print parameter landscape for p = 1
                # gamma
                X = np.linspace(start_gamma, stop_gamma, num_points_gamma)
                # beta
                Y = np.linspace(start_beta, stop_beta, num_points_beta)
                Z = np.zeros((num_points_gamma, num_points_beta))
                for m,x in enumerate(X):
                    for l,y in enumerate(Y):
                        isa_hamiltonian = H_cost.apply_layout(qaoa_circuit.layout)
                        pub = (qaoa_circuit, isa_hamiltonian, [x,y])
                        job = estimator.run(pubs=[pub])
                        primitive_result = job.result()
                        primitive_result = primitive_result[0]
                        # Expectation values of the Hamiltonian
                        Z[m,l] = primitive_result.data.evs
                        #print("Z[m,l]: ", Z[m,l])
                        #Z[m,l] = expectation_value_hamiltonian([x,y], qaoa_circuit, backend, H_cost)

                # Find best Z
                best_i = np.unravel_index(Z.argmin(), Z.shape)

                # save all the info so we can plot it later
                np.save(save_dir + name + "_Z.npy", Z)
                np.save(save_dir + name + "_X.npy", X)
                np.save(save_dir + name + "_Y.npy", Y)
                np.save(save_dir + name + "_best_i.npy", best_i)
                print("best_i: ", best_i)
                np.save(save_dir + name + "_info_dict.npy", info_dict)

            
        else:# load
            Z = np.load(save_dir + name + "_Z.npy")
            X = np.load(save_dir + name + "_X.npy")
            Y = np.load(save_dir + name + "_Y.npy")
            best_i = np.load(save_dir + name + "_best_i.npy")
            best_i = (int(best_i[0]), int(best_i[1]))
            info_dict = np.load(save_dir + name + "_info_dict.npy", allow_pickle=True).item()


        if plot:
            fig = plt.figure(fig_num, figsize=(8, 8), constrained_layout=False)
            fig_num += 1
            font_size = 27
            plt.imshow(Z, cmap="coolwarm", origin="lower", aspect="auto", extent=[Y[0], Y[-1], X[0], X[-1]])
            plt.xlabel(r"$\beta$ (mixer parameter)", fontsize=font_size)
            plt.ylabel(r"$\gamma$ (cost parameter)", fontsize=font_size)
            plt.scatter(Y[best_i[1]], X[best_i[0]], c="black", marker="*", s=1000, label="best params")
            plt.legend(fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.savefig(save_dir + name + '_flat_num_gamma' + str(num_points_gamma) + '.pdf', bbox_inches='tight')
            plt.savefig(save_dir + name + '_flat_num_gamma' + str(num_points_gamma) + '.png')



#ax = fig.add_subplot(projection="2d")
#xx, yy = np.meshgrid(X, Y, indexing='ij')
#surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)
#ax.set_xlabel(r"$\gamma$ (cost parameter)", fontsize=13)
#ax.set_ylabel(r"$\beta$ (mixer parameter)", fontsize=13)

#ax.zaxis.set_label_coords(-1,1)
#ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
#ax.plot(X[best_i[0]], Y[best_i[1]], Z[best_i], c="black", marker="*", markersize = 15, label="best params", zorder=14)
#plt.legend()
#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)

#ax.view_init(azim=0, elev=90)
#plt.savefig(save_dir + name + '_above_num_gamma' + str(num_points_gamma) + '.pdf')
#plt.savefig(save_dir + name + '_above_num_gamma' + str(num_points_gamma) + '.png')
