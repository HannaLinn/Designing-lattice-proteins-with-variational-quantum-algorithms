# Dataset
import networkx as nx


def design_problem(structure: int) -> nx.Graph:
    G = nx.Graph()  # target structure
    nh = 0
    nhlist = [0]

    if structure == 1:
        nh = 2
        nhlist = [2]
        energy=[-1]
        sol=['1001']
        number_nodes = 4
        G.add_nodes_from([i for i in range(0, number_nodes)])
        G.add_edge(0, 3)
    	
    elif structure == 2:
        number_nodes = 8
        nh = 4
        nhlist = [4]
        G.add_nodes_from([i for i in range(0, number_nodes)])
        G.add_edge(0, 7)
        G.add_edge(2, 7)
        G.add_edge(4, 7)
        energy=[-3]
        sol=['10101001']
    elif structure == 3:
        number_nodes = 10
        nh = 4
        nhlist = [4]
        G.add_nodes_from([i for i in range(0, number_nodes)])
        G.add_edge(0, 9)
        G.add_edge(0, 3)
        G.add_edge(3, 6)
        G.add_edge(6, 9)
        energy=[-4]
        sol=['1001001001']
    elif structure == 4:
        number_nodes = 11
        nh = 5
        nhlist = [5]
        G.add_nodes_from([i for i in range(0, number_nodes)])

        G.add_edge(1, 10)
        G.add_edge(1, 8)
        G.add_edge(3, 8)
        G.add_edge(3, 6)
        energy=[-4]
        sol=['10101001010']
    elif structure == 5:
        number_nodes = 12
        nh = 4
        nhlist = [4]
        G.add_nodes_from([i for i in range(0, number_nodes)])

        G.add_edge(0, 11)
        G.add_edge(1, 10)
        G.add_edge(1, 4)
        G.add_edge(4, 7)
        G.add_edge(7, 10)
        energy=[-4]
        sol=['010010010010']
    elif structure == 6:
        number_nodes = 13
        nh = 8
        nhlist = [8, 6]
        G.add_nodes_from([i for i in range(0, number_nodes)])

        G.add_edge(1, 12)
        G.add_edge(3, 12)
        G.add_edge(8, 11)
        G.add_edge(5, 12)
        G.add_edge(6, 1)
        G.add_edge(1, 10)
        energy=[-6]
        sol=['0101011010111']
    elif structure == 7:
        nh = 8
        nhlist = [8]
        number_nodes = 14
        G.add_nodes_from([i for i in range(0, number_nodes)])
        G.add_edge(0, 13)
        G.add_edge(2, 13)
        G.add_edge(4, 13)
        G.add_edge(7, 12)
        G.add_edge(4, 7)
        G.add_edge(9, 12)
        G.add_edge(11, 0)
        energy=[-7]
        sol=['10101001010111']

    elif structure == 8:
        number_nodes = 15
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 8
        nhlist = [8]
        G.add_edge(1, 14)
        G.add_edge(3, 14)
        G.add_edge(5, 14)
        G.add_edge(8, 13)
        G.add_edge(5, 8)
        G.add_edge(10, 13)
        G.add_edge(12, 1)
        energy=[-7]
        sol=['101010010101110']

    elif structure == 9:
        number_nodes = 16
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 6
        nhlist = [6]
        G.add_edge(0, 7)
        G.add_edge(0, 9)
        G.add_edge(0, 11)
        G.add_edge(1, 4)
        G.add_edge(1, 14)
        G.add_edge(2, 15)
        G.add_edge(4, 7)
        G.add_edge(11, 14)
        energy = [-6]
        sol = ["1100100100010010"]

    elif structure == 10:
        number_nodes = 17
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 6
        nhlist = [6]
        G.add_edge(0, 15)
        G.add_edge(0, 13)
        G.add_edge(0, 7)
        G.add_edge(1, 4)
        G.add_edge(1, 6)
        G.add_edge(7, 10)
        G.add_edge(10, 13)
        G.add_edge(15, 2)
        energy=[-6]
        sol=['10100001001001010']

    elif structure == 11:
        number_nodes = 18
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 8
        nhlist = [8, 9]
        G.add_edge(0, 17)
        G.add_edge(0, 15)
        G.add_edge(0, 13)
        G.add_edge(10, 13)
        G.add_edge(1, 10)
        G.add_edge(1, 8)
        G.add_edge(2, 17)
        G.add_edge(2, 5)
        G.add_edge(5, 8)
        energy=[-8]
        sol=['110001001010010001']

    elif structure == 12:
        number_nodes = 19
        nh = 8
        nhlist = [8, 9]
        G.add_nodes_from([i for i in range(0, number_nodes)])
        G.add_edge(0, 17)
        G.add_edge(0, 15)
        G.add_edge(0, 13)
        G.add_edge(10, 1)
        G.add_edge(1, 8)
        G.add_edge(2, 5)
        G.add_edge(2, 17)
        G.add_edge(3, 18)
        G.add_edge(5, 8)
        G.add_edge(10, 13)
        energy=[-8]
        sol=['1110010010100100010']

    elif structure == 13:
        number_nodes = 20
        nh = 8
        nhlist = [8, 9, 11]
        G.add_nodes_from([i for i in range(0, number_nodes)])
        G.add_edge(0, 19)
        G.add_edge(0, 17)
        G.add_edge(0, 15)
        G.add_edge(15, 12)
        G.add_edge(12, 1)
        G.add_edge(1, 10)
        G.add_edge(10, 7)
        G.add_edge(7, 2)
        G.add_edge(6, 3)
        G.add_edge(2, 19)
        energy = [-8]
        sol = ["11100001001010010001"]

    # 30
    elif structure ==14:
        number_nodes=21
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 10
        nhlist=[nh]
        G.add_edge(0,17)
        G.add_edge(1,18)
        G.add_edge(1,20)
        G.add_edge(2,5)
        G.add_edge(5,20)
        G.add_edge(7,20)
        G.add_edge(7,10)
        G.add_edge(10,19)
        G.add_edge(12,19)
        G.add_edge(12,15)
        G.add_edge(15,18)
        sol=['011001010010100100111']
        energy = [-10]
    elif structure ==15:
        number_nodes=22
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 10
        nhlist=[nh]
        G.add_edge(0,21)
        G.add_edge(0,3)
        G.add_edge(0,19)
        G.add_edge(1,18)
        G.add_edge(3,6)
        G.add_edge(6,21)
        G.add_edge(8,21)
        G.add_edge(8,11)
        G.add_edge(11,20)
        G.add_edge(13,20)
        G.add_edge(13,16)
        G.add_edge(16,19)
        sol=['1001001010010100100111']
        energy = [-11]
    elif structure ==16:
        number_nodes=23
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 10
        nhlist=[nh]
        G.add_edge(0,21)
        G.add_edge(1,20)
        G.add_edge(1,4)
        G.add_edge(4,19)
        G.add_edge(6,19)
        G.add_edge(6,9)
        G.add_edge(9,18)
        G.add_edge(11,18)
        G.add_edge(11,14)
        G.add_edge(14,17)
        G.add_edge(17,20)
        G.add_edge(21,16)
        sol=['00111100100101001010010']
        energy = [-10]
    elif structure ==17:
        number_nodes=24
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 10
        nhlist=[nh]
        G.add_edge(0,23)
        G.add_edge(0,5)
        G.add_edge(0,21)
        G.add_edge(1,4)
        G.add_edge(1,20)
        G.add_edge(5,8)
        G.add_edge(8,23)
        G.add_edge(10,23)
        G.add_edge(10,13)
        G.add_edge(13,22)
        G.add_edge(15,18)
        G.add_edge(15,22)
        G.add_edge(18,21)
        sol=['100010010100101001001111']
        energy=[-11]
    elif structure ==18:
        number_nodes=25
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 13
        nhlist=[nh]
        G.add_edge(0,19)
        G.add_edge(1,20)
        G.add_edge(1,22)
        G.add_edge(3,24)
        G.add_edge(3,22)
        G.add_edge(5,24)
        G.add_edge(7,24)
        G.add_edge(7,10)
        G.add_edge(10,23)
        G.add_edge(12,23)
        G.add_edge(12,21)
        G.add_edge(14,17)
        G.add_edge(14,21)
        G.add_edge(17,20)
        sol=['0101010100101010010011111']
        energy = [-13]
    elif structure ==19:
        number_nodes=26
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 14
        nhlist=[nh]
        G.add_edge(0,25)
        G.add_edge(0,17)
        G.add_edge(0,15)
        G.add_edge(1,12)
        G.add_edge(1,10)
        G.add_edge(2,25)
        G.add_edge(2,7)
        G.add_edge(3,24)
        G.add_edge(3,6)
        G.add_edge(7,10)
        G.add_edge(12,21)
        G.add_edge(17,20)
        G.add_edge(20,25)
        G.add_edge(21,24)
        G.add_edge(4,23)
        sol=['11110011001010010100110011']
        energy=[-14]
    elif structure ==20:
        number_nodes=27
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 13
        nhlist=[nh]
        G.add_edge(0,25)
        G.add_edge(0,17)
        G.add_edge(0,15)
        G.add_edge(1,12)
        G.add_edge(1,10)
        G.add_edge(2,25)
        G.add_edge(2,7)
        G.add_edge(3,24)
        G.add_edge(3,6)
        G.add_edge(7,10)
        G.add_edge(12,21)
        G.add_edge(17,20)
        G.add_edge(20,25)
        G.add_edge(21,24)
        G.add_edge(4,23)
        sol=['010101010010101001000011111']
        energy=[-13]
    elif structure ==21:
        number_nodes=28
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 13
        nhlist=[nh]
        G.add_edge(0,5)
        G.add_edge(1,4)
        G.add_edge(1,24)
        G.add_edge(2,27)
        G.add_edge(2,25)
        G.add_edge(3,10)
        G.add_edge(3,12)
        G.add_edge(4,7)
        G.add_edge(7,10)
        G.add_edge(12,27)
        G.add_edge(14,17)
        G.add_edge(14,27)
        G.add_edge(17,26)
        G.add_edge(19,25)
        G.add_edge(19,22)
        G.add_edge(22,25)
        sol=['0011100100101010010100100111']
        energy=[-13]
    elif structure ==22:
        number_nodes=29
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 15
        nhlist=[nh]
        G.add_edge(0,25)
        G.add_edge(1, 26)
        G.add_edge(1, 28)
        G.add_edge(3, 28)
        G.add_edge(5, 28)
        G.add_edge(5, 8)
        G.add_edge(8, 27)
        G.add_edge(9, 22)
        G.add_edge(9, 12)
        G.add_edge(12, 21)
        G.add_edge(14, 21)
        G.add_edge(14, 17)
        G.add_edge(17, 20)
        G.add_edge(20, 23)
        G.add_edge(23, 26)
        G.add_edge(19, 24)
        G.add_edge(24, 27)
        G.add_edge(7, 10)
        sol=['01010100110010100100111100111']
        energy=[-15]   
    elif structure == 23:
        number_nodes = 30
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 15
        nhlist = [15, 14, 16, 13, 17]
        G.add_edge(0, 5)
        G.add_edge(0, 7)
        G.add_edge(0, 9)
        G.add_edge(1, 12)
        G.add_edge(1, 26)
        G.add_edge(2, 27)
        G.add_edge(2, 5)
        G.add_edge(9, 12)
        G.add_edge(11, 14)
        G.add_edge(13, 26)
        G.add_edge(13, 16)
        G.add_edge(16, 25)
        G.add_edge(18, 21)
        G.add_edge(18, 25)
        G.add_edge(21, 24)
        G.add_edge(24, 27)
        G.add_edge(23, 28)
        G.add_edge(3, 28)
        sol=['111001010100110010100100111100']
        energy=[-15]
    elif structure == 24:
        number_nodes = 48
        G.add_nodes_from([i for i in range(0, number_nodes)])

        G.add_edge(0, 3)
        G.add_edge(2, 5)
        G.add_edge(2, 21)
        G.add_edge(4, 7)
        G.add_edge(5, 20)
        G.add_edge(6, 19)
        G.add_edge(6, 9)
        G.add_edge(9, 18)
        G.add_edge(10, 17)
        G.add_edge(15, 42)
        G.add_edge(16, 47)
        G.add_edge(16, 43)
        G.add_edge(18, 47)
        G.add_edge(19, 46)
        G.add_edge(20, 23)
        G.add_edge(23, 46)
        G.add_edge(24, 45)
        G.add_edge(22, 25)
        G.add_edge(25, 32)
        G.add_edge(27, 32)
        G.add_edge(28, 31)
        G.add_edge(31, 34)
        G.add_edge(33, 36)
        G.add_edge(35, 38)
        G.add_edge(36, 45)
        G.add_edge(37, 44)
        G.add_edge(37, 40)
        G.add_edge(40, 43)
        G.add_edge(24, 33)
        G.add_edge(44, 47)
        sol=[]
        energy=[]
    elif structure == 25:
        number_nodes = 50
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 31
        nhlist = [31]
        G.add_edge(0, 17)
        G.add_edge(0, 19)
        G.add_edge(0, 23)
        G.add_edge(1, 8)
        G.add_edge(1, 6)
        G.add_edge(2, 23)
        G.add_edge(2, 5)
        G.add_edge(3, 24)
        G.add_edge(3, 26)
        G.add_edge(4, 27)
        G.add_edge(4, 33)
        G.add_edge(5, 34)
        G.add_edge(6, 49)
        G.add_edge(7, 48)
        G.add_edge(7, 46)
        G.add_edge(8, 17)
        G.add_edge(9, 16)
        G.add_edge(10, 13)
        G.add_edge(11, 44)
        G.add_edge(9, 46)
        G.add_edge(10, 45)
        G.add_edge(13, 16)
        G.add_edge(15, 18)
        G.add_edge(19, 22)
        G.add_edge(27, 30)
        G.add_edge(34, 49)
        G.add_edge(32, 35)
        G.add_edge(30, 33)
        G.add_edge(35, 38)
        G.add_edge(38, 49)
        G.add_edge(39, 48)
        G.add_edge(40, 47)
        G.add_edge(42, 45)
        G.add_edge(42, 47)
        sol=[]
        energy=[]
    elif structure == 26:
        number_nodes = 52
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nhlist = [31]

        G.add_edge(0, 17)
        G.add_edge(0, 19)
        G.add_edge(0, 23)
        G.add_edge(1, 8)
        G.add_edge(1, 6)
        G.add_edge(2, 23)
        G.add_edge(2, 5)
        G.add_edge(3, 24)
        G.add_edge(3, 26)
        G.add_edge(4, 27)
        G.add_edge(4, 33)
        G.add_edge(5, 34)
        G.add_edge(6, 51)
        G.add_edge(7, 50)
        G.add_edge(7, 48)
        G.add_edge(8, 17)
        G.add_edge(9, 16)
        G.add_edge(10, 13)
        G.add_edge(11, 46)
        G.add_edge(9, 48)
        G.add_edge(10, 47)
        G.add_edge(13, 16)
        G.add_edge(15, 18)
        G.add_edge(19, 22)
        G.add_edge(27, 30)
        G.add_edge(30, 33)
        G.add_edge(34, 51)
        G.add_edge(32, 35)
        G.add_edge(35, 38)
        G.add_edge(38, 51)
        G.add_edge(39, 50)
        G.add_edge(39, 42)
        G.add_edge(44, 49)
        G.add_edge(44, 47)
        G.add_edge(42, 49)
        G.add_edge(37, 40)
        sol=[]
        energy=[-32]
    elif structure == 27:
        number_nodes = 64
        G.add_nodes_from([i for i in range(0, number_nodes)])
        nh = 38
        nhlist = [38, 42]
        G.add_edge(0, 9)
        G.add_edge(0, 13)
        G.add_edge(0, 15)
        G.add_edge(1, 18)
        G.add_edge(1, 8)
        G.add_edge(2, 7)
        G.add_edge(2, 19)
        G.add_edge(3, 6)
        G.add_edge(3, 22)
        G.add_edge(4, 29)
        G.add_edge(4, 23)
        G.add_edge(5, 30)
        G.add_edge(5, 58)
        G.add_edge(6, 57)
        G.add_edge(7, 56)
        G.add_edge(8, 55)
        G.add_edge(9, 54)
        G.add_edge(10, 13)
        G.add_edge(10, 53)
        G.add_edge(11, 52)
        G.add_edge(15, 18)
        G.add_edge(17, 20)
        G.add_edge(19, 22)
        G.add_edge(21, 24)
        G.add_edge(23, 26)
        G.add_edge(26, 29)
        G.add_edge(28, 31)
        G.add_edge(30, 33)
        G.add_edge(32, 35)
        G.add_edge(33, 58)
        G.add_edge(34, 59)
        G.add_edge(34, 37)
        G.add_edge(37, 40)
        G.add_edge(39, 42)
        G.add_edge(40, 59)
        G.add_edge(41, 44)
        G.add_edge(41, 60)
        G.add_edge(43, 46)
        G.add_edge(44, 61)
        G.add_edge(45, 62)
        G.add_edge(45, 48)
        G.add_edge(48, 63)
        G.add_edge(50, 63)
        G.add_edge(50, 53)
        G.add_edge(54, 63)
        G.add_edge(55, 62)
        G.add_edge(56, 61)
        G.add_edge(57, 60)
        sol=[]
        energy=[]

    else:
        print("No structure number was given.")

    return (G, nhlist,energy,sol)


def set_of_design_problems(n: int, all: bool = False) -> list[nx.Graph]:
    """
    Input: n = number of design problems, max 6
    Output: list of problem of length n
    """
    if all:
        n = 23
    return_list = []
    for structure in range(1, n + 1):
        return_list.append(design_problem(structure))

    return return_list
