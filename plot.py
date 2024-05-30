import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import time
from pathlib import Path
#from matplotlib import rc
#rc('text', usetex=True)

def best_fitting_line(x):
    # return 0.0290974 * x + 467.301
    return 0.290974 * x + 467.01

def plotMenchmarkResultLjDirectSummation():
    time = [6.7752e-3, 80.491e-3, 561.25e-3, 2.3192, 6.5419, 17.696, 39.387, 75.713, 142.85]
    atoms = [8, 27, 64, 125, 216, 343, 512, 729, 1000]
    plt.plot(time,atoms, marker='x',label='Rust')
    time = [7.2057e-2,0.854074, 4.8465, 19.8629, 58.731, 150.404, 304.998, 567.507, 998.639]
    plt.plot(time,atoms, marker='x',label='C++')
    plt.title("LJ Direct Summation Benchmark")
    plt.xlabel('elapsed time in ms')
    plt.ylabel('number of atoms')
    plt.legend()
    plt.grid()
    plt.show()

def readAndPlotMilestone08():
    # data = np.loadtxt(args.filename)
    # print(type(data))
    # print(data)
    e_tot_x=[]
    e_tot_y=[]
    e_pot_x=[]
    e_pot_y=[]
    e_kin_x=[]
    e_kin_y=[]
    #the working dir should be the build/milestones/04/
    script_path = os.path.dirname(os.path.abspath(__file__))

    datareader =  csv.reader(open("output_milestone07_e_tot_923.csv"),delimiter=';')
    for row in datareader:
        e_tot_y.append(float(row[0]))
        e_tot_x.append(float(row[1]))

    datareader =  csv.reader(open("output_milestone07_e_pot_923.csv"),delimiter=';')
    for row in datareader:
        e_pot_y.append(float(row[0]))
        e_pot_x.append(float(row[1]))
    
    datareader =  csv.reader(open("output_milestone07_e_kin_923.csv"),delimiter=';')
    for row in datareader:
        e_kin_y.append(float(row[0]))
        e_kin_x.append(float(row[1]))
    
    #print("X:",x)
    #print("\n\n")
    #print("Y",y)

    #Plot relation of E_kin and E_pot
    plt.figure(1)
    plt.title("$E_{kin}$ vs. $E_{pot}$")
    plt.plot(e_pot_x,e_pot_y,color='tab:blue',label='$E_{pot}$')
    plt.ylabel('$E_{pot}$ in $Ev$',color='tab:blue')
    plt.xlabel('Time $t$ in $fs$')
    E_kin_y_axis = plt.twinx()
    E_kin_y_axis.plot(e_kin_x,e_kin_y,color='tab:red')
    
    plt.ylabel('$E_{kin}$ in $Ev$',color='tab:red')
    plt.show()

    plt.figure(2)
    e_tot_color='tab:green'
    plt.title("$E_{tot}$ Consistency Parallelized")
    plt.plot(e_tot_x,e_tot_y,label='$E_{tot}$',color=e_tot_color)
    plt.ylabel('$E_{tot}$ in $Ev$')
    plt.xlabel('Time $t$ in $fs$')
    # plt.ylim(-0.725,-0.52)
    plt.show()

plotMenchmarkResultLjDirectSummation()
# plotTest()


# parser = argparse.ArgumentParser(
#     prog='plot.py',
#     description='plot the given data of a specific milestone'
# )

# parser.add_argument('filename')
# parser.add_argument('milestone')

# args = parser.parse_args()

# if args.milestone==str(4):
#     readAndPlotMilestone04()
# else:
#     print("milestone not correct!!")





