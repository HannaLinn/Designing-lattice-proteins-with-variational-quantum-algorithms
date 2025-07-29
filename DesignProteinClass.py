"""
HP-lattice Protein Design
Based on Lucas Knuthsons code
"""

import numpy as np
from collections import defaultdict
from itertools import combinations, product
import networkx as nx
from typing import Tuple


class DesignProteinInstance:
    """
    Class for one instance of the 2D lattice HP-model to then be fed into the quantum simulation.

    Class variables in order of creation [type]:
    After init:
            - structure = Graph
            - lamda = Lagrange parameter
            - Q = dict with energy info [dictionary]
            - num_bits = number of bits in the instance (same as nodes in the structure) [int]
            - O_energies = one-body energies [list of floats]
            - T_energies = two-body energies [list of Numpy Arrays with floats]

    After function call:
            - feasible_set = Numpy arrays of all feasible bitstrings solutions [list of Numpy Arrays]
            - solution_set = Numpy arrays of all bitstring solutions [list of Numpy Arrays]

    """

    def __init__(self, structure: nx.Graph, NH: int, lamda: float = 1.5):
        self.structure = structure
        self.lamda = lamda  # best lamda: 1.5
        self.NH = NH
        self.Q = self.make_Q(structure)

        self.bit_name_list = self.get_bit_names()
        self.num_bits = len(self.bit_name_list)
        self.O_energies = self.get_O_energies()
        self.T_energies = self.get_T_energies()

    def __str__(self) -> str:
        return (
            "\nO:\n"
            + str(self.O_energies)
            + "\nT:\n"
            + str(self.T_energies)
            + "\nLamda:\n"
            + str(self.lamda)
        )

    def make_Q(
        self, G: nx.Graph, verbose: bool = False
    ) -> dict[tuple[int, int], float]:
        """
        Q is the interactions in the Ising model.
        Two-body energy: (q_1, q_2) = value
        One-body energy: (q_1, q_1) = value

        Bit format:
        Based on code by: Lucas Knuthson
        """

        Q = defaultdict(int)  # type: dict[tuple[int, int], float]

        ####### Energy function: E_HP(C_t, s) #######
        # w_ij s_i s_j
        for u, v in G.edges:
            Q[(u, v)] += -1.0

        ####### Penalty part of energy function #######
        #  biases the total number of H beads toward a preset value
        for node in G.nodes:
            Q[(node, node)] += (1.0 - 2.0 * self.NH) * self.lamda

        for x, y in combinations(G.nodes, 2):
            Q[(x, y)] += 2.0 * self.lamda

        if verbose:
            print(Q)

        Q = dict(Q)  # not a defaultdict anymore to not be able to grow by error
        return Q

    def get_bit_names(self) -> list[int]:
        return list(self.structure.nodes())

    def get_O_energies(self) -> list[float]:
        """
        Get the one-body energies for the Hamiltonian.
        """
        O_energies = []
        for bit in self.bit_name_list:
            try:
                O_energies.append(self.Q[(bit, bit)])
            except:
                pass
        return O_energies

    def get_T_energies(self) -> np.ndarray:
        """
        Get the two-body energies for the Hamiltonian.
        """
        T_energies = np.zeros((self.num_bits, self.num_bits))

        for j in range(self.num_bits):
            for k in range(self.num_bits):
                if j == k:
                    T_energies[j, k] = 0
                else:
                    try:
                        T_energies[j, k] = self.Q[
                            self.bit_name_list[j], self.bit_name_list[k]
                        ]
                        if j > k:
                            T_energies[k, j] = self.Q[
                                self.bit_name_list[j], self.bit_name_list[k]
                            ]
                    except:
                        pass

        T_energies = np.triu(T_energies)  # delete lower triangle
        T_energies = (
            T_energies + T_energies.T - np.diag(np.diag(T_energies))
        )  # copy upper triangle to lower triangle
        return T_energies

    def get_feasible_percentage(self) -> float:
        return 100 * (len(self.feasible_set) / len(self.solution_set))

    def get_solution_set(self) -> list[np.ndarray]:
        """
        Input: Number of bits.
        Output: Numpy arrays of dimensions (1, num_bits) in a list of all possible bitstrings.
        """
        return [np.array(i) for i in product([0, 1], repeat=self.num_bits)]

    def get_feasible_set(self) -> list[np.ndarray]:
        """
        Generate ordered binary strings of length num_bits with Hamming weight NH
        """
        self.solution_set = self.get_solution_set()

        if self.NH > self.num_bits:
            raise ValueError("NH should be less than or equal to the number of bits.")

        """
        # Generate all possible combinations of indices for '1's in the binary string
        indices_combinations = combinations(range(self.num_bits), self.NH)
    
        # Initialize an empty list to store the generated binary strings
        binary_strings = []
    
        # Iterate through each combination of indices
        for indices in indices_combinations:
            # Create a binary string with '1' at the specified indices
            binary_string = ['0'] * self.num_bits
            for index in indices:
                binary_string[index] = '1'
        
            # Append the generated binary string to the list
            binary_strings.append(''.join(binary_string))
    
        return binary_strings
    
        """
        self.solution_set = self.get_solution_set()
        return_list = []
        for sol in self.solution_set:
            if sum(sol) == self.NH:
                return_list.append(sol)
        return return_list

    def calc_solution_sets(self):
        self.feasible_set = self.get_feasible_set()

    def bit2energy(self, bit_array) -> float:
        """
        Returns the classical energy corresponding to the given bitstring.
        """
        Oe = np.dot(bit_array, self.O_energies)
        Te = 0
        for j, bit in enumerate(self.bit_name_list):
            for k, bit in enumerate(self.bit_name_list):
                if bit_array[j] == 1.0 and bit_array[k] == 1.0:
                    Te += self.T_energies[j, k]
        energy = Oe + Te
        return energy

    def energy_of_set(
        self, feasible: bool = False, verbose: bool = False
    ) -> Tuple[list[float], list[str], list[np.ndarray | int | float]]:
        """
        Returns the classical energy corresponding to the given bitstrings.
            - energy_list = list of the classical energy for given set of solutions
            - labels = list of the strings with the bitstrings for given set of solutions
            - lowest_energy_bitstring = [np.array of best bitstring, index in feasible set, classical energy]
        """
        energy_list = []
        labels = []
        mem = 1000000.0
        lowest_energy_bitstring = None
        if feasible:
            set_ = self.feasible_set
        else:
            set_ = self.solution_set
        for i in range(len(set_)):
            energy = self.bit2energy(set_[i])

            if verbose and (i % 1000 == 0):
                print(
                    "Progress in energy calculations: ",
                    round(100 * i / len(set_), 1),
                    "%%",
                )
            try:
                energy = self.bit2energy(set_[i])
                if energy < mem:
                    lowest_energy_bitstring = [set_[i], i, energy]
                    mem = energy
                label = str(set_[i])
                label = label.replace(",", "")
                label = label.replace(" ", "")
                label = label.replace(".", "")
                labels.append(label)
            except:
                energy = 1000000
                if not feasible:
                    label = str(set_[i])
                    label = label.replace(",", "")
                    label = label.replace(" ", "")
                    label = label.replace(".", "")
                    labels.append(label)
            energy_list.append(energy)
        if verbose:
            print("Done!")
        return energy_list, labels, lowest_energy_bitstring
