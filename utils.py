import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib
from datetime import datetime
import numpy as np
from scipy.io import savemat
from collections import defaultdict
import ast, re

# TODO:
# - typing

def save_constants_to_file(file_path: str, constants_dict: dict) -> None:
    with open(file_path, "w") as f:
        for key, value in constants_dict.items():
            f.write(f"{key}: {value}\n")

def load_constants_from_file(file_path: str) -> dict:
    constants_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.split(": ")
            constants_dict[key] = value
    return constants_dict

def parse_constants(constant_dict):
    # dataset is a string that contains a list of tuples with lists in it
    dataset_string = constant_dict["data_set"]
    dataset_string = re.sub(r'<[^>]*>', '[]', dataset_string)
    dataset = ast.literal_eval(dataset_string)
    simulate = ast.literal_eval(constant_dict["simulate"])
    noisy_simulation = ast.literal_eval(constant_dict["noisy_simulation"])
    maxiter = ast.literal_eval(constant_dict["maxiter"])
    average_iter = ast.literal_eval(constant_dict["average_iter"])
    num_layers = ast.literal_eval(constant_dict["num_layers"])
    lamda = ast.literal_eval(constant_dict["lamda"])
    default_shots = ast.literal_eval(constant_dict["default_shots"])

    # because somthing is wrong with dataset!!
    if len(dataset[7][-1][0]) == 8:
        dataset[7][-1][0] = dataset[7][-1][0] + '0000000'
    return dataset, simulate, noisy_simulation, maxiter, average_iter, num_layers, lamda, default_shots


def read_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

def plot_cost(
    cost_vector,
    name,
    pdf: bool = False,
    new_fig: bool = True,
    ground_energy: float = None,
    label: str = "Cost",
    color: str = None,
):
    if new_fig:
        plt.figure()
    plt.plot(cost_vector, label=label, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    if ground_energy is not None:
        plt.plot(
            np.full(len(cost_vector), ground_energy),
            label=label + " ground energy",
            color=color,
        )
    plt.legend()
    if pdf:
        plt.savefig(name + ".pdf", bbox_inches="tight")
    else:
        print("saving as jpg")
        plt.savefig(name + ".jpg")

def sort_over_threshhold(x, threshhold: float):
    mask = x > threshhold
    sort_idx = x.argsort()
    mask_idx = sort_idx[mask[sort_idx]]
    return x[mask_idx], mask_idx


def plot_probs_with_energy(
    probs,
    num_qubits,
    H_cost,
    ground_states_i,
    new_fig=True,
    save=False,
    name="",
    threshhold=0.001,
):
    """
    Plots given probailities with the corresponding bitstrings.
    """

    x, mask_idx = sort_over_threshhold(probs, threshhold)  # high enough probabilities
    if len(x) < 1:
        print("No solutions with a probability over the given threshhold: ", threshhold)

    indices = np.arange(len(probs), dtype=int)[mask_idx]
    labels_y = index_set2bit_string(indices, num_qubits)  # bit strings

    y = np.array(
        [x for x in energies_of_set(indices, H_cost, num_qubits)]
    )  # energies of the set

    if new_fig:
        plt.figure(np.random.randint(10, 30))

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm", len(y))
    scat = ax.scatter(y, y, c=y, cmap=cmap)  # get the color of the sidebar correct
    plt.cla()  # clear plot
    cbar = plt.colorbar(scat, ax=plt.gca())  # colorbar on the side
    cbar.set_label("Classic energy", labelpad=15, fontsize=20)

    norm = matplotlib.colors.Normalize(vmin=np.min(y), vmax=np.max(y))
    norms = norm(y)

    colors = matplotlib.cm.coolwarm(norms)  # get the colors of the columns
    ax.bar(
        range(len(x)), x, color=colors
    )  # plot the columns with the probabilities as the hight of the bar, colors are classical energy
    plt.xticks(ticks=range(len(y)), labels=labels_y)  # bit strings as labels

    # mark out the best bit in green
    for i in range(len(y)):
        if round(float(y[i]), 4) == round(
            energy_of_index(ground_states_i[0], H_cost), 4
        ):
            ax.get_xticklabels()[i].set_color("green")

    plt.ylabel("Probability", fontsize=20)
    plt.title(r"Probability of measuring bit strings with energy", fontsize=25)
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=17)
    plt.xticks(rotation=85)

    if save:
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y")
        plt.savefig(name + "_probs_with_energy_" + date_time + ".pdf")


# Energy functions


def energies_of_set(_set, H_cost, num_qubits):
    """
    Set can be indices or arrays.
    """
    try:
        if len(_set[0]) >= num_qubits:
            indices = bit_array_set2indices(_set)
    except:
        indices = _set
    energies_index_states = get_eigenvalues(H_cost)
    energies = np.take(energies_index_states, indices)
    return energies


def get_eigenvalues(H_cost, verbose=False):
    if verbose:
        print("\nCalculating energy of states")
    energy_list = []
    matrix = H_cost.to_matrix()
    return matrix.diagonal().real.round(
        8
    )  # some weird thing at the conversion demands a round, it should not matter at 8th decimal


def energy_of_index(index, H_cost):
    energies_index_states = get_eigenvalues(H_cost)
    return energies_index_states[index]


# Ground states


def get_ground_states_i(feasible_set, H_cost):
    indices_of_feasible = bit_array_set2indices(feasible_set)
    energies_of_feasible = energies_of_set(
        indices_of_feasible, H_cost, len(feasible_set[0])
    )
    ground_energy = round(
        float(np.amin(energies_of_feasible)), 8
    )  # just to avoid a weird 4 at the 20th decimal
    ground_states_i = np.take(
        indices_of_feasible, np.where(energies_of_feasible <= ground_energy)
    )[0]
    return ground_energy, ground_states_i


# Transforms: strings <-> bit_array


def bit_array_set2indices(bit_array_set):
    return np.array(list(map(bit_array2index, bit_array_set)))


def bit_array2index(bit_array):
    return int(
        (
            bit_array
            * (2 ** np.linspace(bit_array.size - 1, 0, bit_array.size, dtype=np.uint64))
        ).sum()
    )


def index_set2bit_string(index_set, num_qubits):
    """
    Input: Indices of a set of bit strings in a list, e.g. [12, 542, 1].
    Output: The whole set of bits in a list, e.g. np array of bit string, e.g. [1000100, 110101, 101010].
    """
    packed_index_set = []
    for i in index_set:
        temp = index2bit_string(i, num_qubits)
        packed_index_set.append(temp)
    return packed_index_set


def bit_array2string(state):
    bit_string = ""
    for c in state:
        if int(c) == 1:
            bit_string += "1"
        elif int(c) == 0:
            bit_string += "0"
    return bit_string


def string2index(string):
    return int(string, 2)


def index2bit_string(index, num_qubits):
    """
    Input: Index in pennylane basis, to be used in e.g. probability vector.
    Output: bit_string of a state as a string, e.g. '1000100'.
    """
    string = bin(index)[2:]
    while len(string) < num_qubits:
        string = "0" + string
    return string


# Grid search
# TODO: rewrite to qiskit --------------------------
# Or use orqid


def grid_search(
    start_gamma,
    stop_gamma,
    num_points_gamma,
    start_beta,
    stop_beta,
    num_points_beta,
    heuristic,
    args=(),
    plot=True,
    matplot_save=False,
    above=False,
    save=False,
    name="",
):
    """
    Calculates the best parameters [gamma, beta], gives the lowest average cost, within the interval given.
    Heuristic neest to return a float.
    """
    # gamma
    X = np.linspace(start_gamma, stop_gamma, num_points_gamma)
    # beta
    Y = np.linspace(start_beta, stop_beta, num_points_beta)

    Z = np.zeros((num_points_gamma, num_points_beta))
    for m, x in enumerate(X):
        for l, y in enumerate(Y):
            Z[m, l] = heuristic([float(x), float(y)], *args)

    # Find best Z
    i = np.unravel_index(Z.argmin(), Z.shape)

    if plot:
        plot_grid_search(X, Y, Z, i, above=above, save=save, name=name)

    if matplot_save:
        mdic = {
            "Z_av": Z,
            "X": np.linspace(start_gamma, stop_gamma, num_points_gamma),
            "Y": np.linspace(start_beta, stop_beta, num_points_beta),
        }
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y")
        savemat("Matlab_" + date_time + str(num_points_gamma) + ".mat", mdic)

    gamma = float(X[i[0]])
    beta = float(Y[i[1]])

    return [gamma, beta], Z, i


def grid_search_p2(
    start_gamma,
    stop_gamma,
    num_points_gamma,
    start_beta,
    stop_beta,
    num_points_beta,
    heuristic,
    args=(),
):
    """
    Calculates the best parameters [gamma, beta] for p=2, gives the lowest cost, within the interval given.
    If vmap is on then JAX and JIT is used for speeding up the calculations.
    """
    # gamma
    X1 = np.linspace(start_gamma, stop_gamma, num_points_gamma)
    X2 = np.linspace(start_gamma, stop_gamma, num_points_gamma)
    # beta
    Y1 = np.linspace(start_beta, stop_beta, num_points_beta)
    Y2 = np.linspace(start_beta, stop_beta, num_points_beta)

    Z = np.zeros((num_points_gamma, num_points_gamma, num_points_beta, num_points_beta))
    for xi, x1 in enumerate(X1):
        for xj, x2 in enumerate(X2):
            for yi, y1 in enumerate(Y1):
                for yj, y2 in enumerate(Y2):
                    Z[xi, xj, yi, yj] = heuristic(
                        [float(x1), float(x2), float(y1), float(y2)], *args
                    )

    # Find best Z
    i = np.unravel_index(Z.argmin(), Z.shape)

    gamma1 = float(X1[i[0]])
    gamma2 = float(X2[i[1]])
    beta1 = float(Y1[i[2]])
    beta2 = float(Y1[i[3]])

    return [gamma1, gamma2, beta1, beta2], Z, i


def get_batch_array(
    start_gamma, stop_gamma, num_points_gamma, start_beta, stop_beta, num_points_beta
):
    # gamma
    X = np.linspace(start_gamma, stop_gamma, num_points_gamma)
    # beta
    Y = np.linspace(start_beta, stop_beta, num_points_beta)
    batch_list = []
    for x in X:
        for y in Y:
            temp = np.array([[float(x)], [float(y)]], dtype=float)
            batch_list.append(temp)

    batch_array = np.array(batch_list, dtype=float)
    return X, Y, batch_array


def plot_grid_search(X, Y, Z, i, above=False, name="", save=False, fontsize=13):
    fig = plt.figure()  #  , constrained_layout=True
    ax = fig.add_subplot(projection="3d")
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    surf = ax.plot_surface(xx, yy, Z, cmap=cm.BrBG, antialiased=False)
    ax.zaxis.set_label_coords(-1, 1)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    ax.plot(X[i[0]], Y[i[1]], Z[i], c="red", marker="*", label="best params", zorder=10)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(r"Best params: $\gamma$ " + str(X[i[0]]) + r", $\beta$ " + str(Y[i[1]]))
    num_points_gamma = len(X)
    ax.set_xlabel(r"$\gamma$ (cost parameter)", fontsize=fontsize)
    ax.set_ylabel(r"$\beta$ (mixer parameter)", fontsize=fontsize)
    ax.set_zlabel(name, fontsize=fontsize)
    if save:
        plt.savefig(name + "_num_gamma" + str(num_points_gamma) + ".pdf")

    if above:
        ax.view_init(azim=0, elev=90)
        if save:
            plt.savefig(name + "_above_num_gamma" + str(num_points_gamma) + ".pdf")


def get_annealing_params(
    annealing_time, p, linear=True, cosine=False, sine=False, save=False, plot=False
):
    """
    From Tutorial 3 AdvQuantumAlgorithms Course.

    Sine function does not work.

    """
    if sum([linear, cosine, sine]) >= 2:
        raise Exception("Choose only one schedule")
    if sum([linear, cosine, sine]) == 0:
        raise Exception("Choose a schedule")

    annealing_params = np.zeros((2, p))
    tau = annealing_time / p
    if linear:
        name = "linear_"
        for i in range(p):
            annealing_params[0, i] = (
                tau * (i + 1 - 0.5) / p
            )  # gamma  Trotterisation to 2nd order
            annealing_params[1, i] = -tau * (1 - ((i + 1) / p))  # beta
        annealing_params[1, p - 1] = -tau / (4 * p)  # Trotterisation to 2nd order

    elif cosine:
        name = "cosine_"
        B_function = lambda s: (np.cos(np.pi + (s) * np.pi) + 1) / 2
        for i in range(p):
            annealing_params[0, i] = tau * B_function((i + 1 - 0.5) / p)
            annealing_params[1, i] = -(tau / 2) * (
                2 - B_function((i + 1 + 0.5) / p) - B_function((i + 1 - 0.5) / p)
            )
        annealing_params[1, p - 1] = -(tau / 2) * (1 - B_function((p - 0.5) / p))

    elif sine:
        name = "sine_"
        B_function = lambda s: np.tan(-np.pi / 2 + (s) * np.pi)
        for i in range(p):
            annealing_params[0, i] = tau * B_function((i + 1 - 0.5) / p)
            annealing_params[1, i] = -(tau / 2) * (
                2 - B_function((i + 1 + 0.5) / p) - B_function((i + 1 - 0.5) / p)
            )
        annealing_params[1, p - 1] = -(tau / 2) * (1 - B_function((p - 0.5) / p))

    if plot and not save:
        raise Exception("Must save the parameters to be able to plot")
    if save:
        np.savetxt(name + "_params" + ".out", annealing_params, delimiter=",")
        if plot:
            plot_params(name, 1, p, save=True)

    return annealing_params.tolist()[0] + annealing_params.tolist()[1]


def interpolate_params(params, only_last=False, save=False, plot=False):
    """
    By Lucin appendix B, p 14
    """
    p = int(len(params) / 2)
    params = np.array([params[0:p], params[p:]])

    params_vector = np.concatenate((params, np.full((2, 1), 0.0)), axis=1)
    params_p_plus1 = np.full((2, p + 1), 0.0)

    for i in range(0, p + 1):
        params_p_plus1[0, i] = ((i - 1 + 1) / p) * params_vector[0, i - 1] + (
            (p - (i + 1) + 1) / p
        ) * params_vector[0, i]
        params_p_plus1[1, i] = ((i - 1 + 1) / p) * params_vector[1, i - 1] + (
            (p - (i + 1) + 1) / p
        ) * params_vector[1, i]

    if only_last:
        i = p
        params_vector[0, i] = ((i - 1 + 1) / p) * params_vector[0, i - 1] + (
            (p - (i + 1) + 1) / p
        ) * params_vector[0, i]
        params_vector[1, i] = ((i - 1 + 1) / p) * params_vector[1, i - 1] + (
            (p - (i + 1) + 1) / p
        ) * params_vector[1, i]
        return_params = params_vector
    else:
        return_params = params_p_plus1

    if plot and not save:
        raise Exception("Must save the parameters to be able to plot")
    if save:
        name = "interpolated"
        np.savetxt(name + "_params" + ".out", return_params, delimiter=",")
        if plot:
            plot_params(name, 1, p + 1, save=True)

    return return_params.tolist()[0] + return_params.tolist()[1]


"""
def init_params(name, p, **kwargs):
    if name == 'zero':
        return [0.0] * 2 * p
    elif name == 'bound_based':
        bounds = kwargs['bounds']
        return [(bound[1] - bound[0]) / 2 for bound in bounds]
    elif name == 'annealing':
        return get_annealing_params(kwargs['annealing_time'], p, kwargs['linear'], kwargs['cosine'], kwargs['sine'])
"""


def plot_params(name, p_min, p_max, color="", color2=False, trained="", save=False):
    params = open(name + "_params" + ".out", "r").readlines()
    names = [r"$\gamma$", r"$\beta$"]
    colors = ["blue", "green"]
    if color2:
        colors = ["coral", "seagreen"]
    markers = ["o", "s"]
    for i, line in enumerate(params):
        line = np.fromstring(line, dtype=float, sep=",")
        plt.plot(
            np.arange(p_min, p_max + 1),
            line,
            label=trained + names[i],
            color=color + colors[i],
            marker=markers[i],
            linestyle="dashed",
            linewidth=2,
            markersize=9,
        )
        plt.legend(fontsize=13)
        plt.xlabel("p", fontsize=19)
        plt.yticks(fontsize=19)
        plt.grid(True)
        plt.xticks(np.linspace(p_min, p_max, 3, dtype=int), fontsize=13)
        plt.ylabel("Radians", fontsize=17)
    if save:
        plt.savefig("plot_params" + name + ".pdf", bbox_inches="tight")


# TODO: check that it works same as np and not jnp
def get_minimal_energy_gap(H_cost):
    get_eigenvalues(H_cost)
    return np.diff(np.unique(energies_index_states)).min()


# save run information
def save_info(info_dict: dict[str, float], file_name: str):
    file2 = open(file_name, "w")
    save_string = ""
    for key in info_dict:
        save_string = save_string + key + ": " + str(info_dict[key]) + "\n"
    file2.writelines(save_string)
    file2.close()


def count_gates(circuit):
    counts = defaultdict()
    counts[1] = 0
    counts[2] = 0
    for inst in circuit.data:
        counts[inst.operation.num_qubits] += 1
    return counts
