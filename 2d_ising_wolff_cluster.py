import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

rng = np.random.default_rng()

parser = argparse.ArgumentParser('2D ising simulation')
parser.add_argument('--melting_iterations', type=int)
parser.add_argument('--measuring_iterations', type=int)
args = parser.parse_args()

def hamiltonian(spins, J=1):
    N = spins.shape[0]
    energy = 0
    
    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            
            right = spins[i, (j + 1) % N]
            down  = spins[(i + 1) % N, j]
            
            energy += -J * s * right
            energy += -J * s * down
            
    return energy

def wolff_cluster_logic(N, T, spins):

    random_site_x = rng.integers(N)
    random_site_y = rng.integers(N)

    random_site = (random_site_x, random_site_y)
    cluster = set()
    cluster.add(random_site)
    f_old = set()
    f_old.add(random_site)

    while len(f_old) > 0:
        f_new = set()
        for test_pair in f_old:
            test_pair_x = test_pair[1]
            test_pair_y = test_pair[0]

            right = (test_pair_y, (test_pair_x + 1)%N)
            down = ((test_pair_y + 1)%N, test_pair_x)
            left = (test_pair_y, (test_pair_x - 1)%N)
            up = ((test_pair_y - 1)%N, test_pair_x)

            neighbors = [up, down, right, left]

            for neighbor in neighbors:

                neighbor_x = neighbor[1]
                neighbor_y = neighbor[0]

                if (neighbor not in cluster) and (spins[neighbor_y,neighbor_x] == spins[test_pair_y, test_pair_x]):

                    B = 1/T
                    if np.random.rand() < (1-np.exp(-2*B)):
                        f_new.add(neighbor)
                        cluster.add(neighbor)
        f_old = f_new
        
    for spin in cluster:
        x = spin[1]
        y = spin[0]
        spins[y,x]*=-1

    return cluster

def wolff_cluster(N, T, spins, melting_iterations, measuring_iterations):
    
    temp_cluster_size_array = []
   
    for i in range(melting_iterations):
        wolff_cluster_logic(N,T,spins) 
    for i in range(measuring_iterations):
        if i == measuring_iterations-1:
            cluster = wolff_cluster_logic(N,T,spins)
            return cluster, spins

        wolff_cluster_logic(N,T,spins)

def temperature_logic(args):
    
    N, T, melting_iterations, measuring_iterations = args

    spins_wolff = np.ones((N,N))

    graphing_cluster, graphing_spins = wolff_cluster(N, T, spins_wolff, melting_iterations, measuring_iterations)

    return T, graphing_cluster, graphing_spins

def run_sim(N, melting_iterations, measuring_iterations):
    
    temperature_array = np.array([1.5, 2.3, 3.5])
    
    args = [(N, T, melting_iterations, measuring_iterations) for T in temperature_array]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(temperature_logic, args), total=len(args)))

    T_arr, cluster_arr, spins_arr = zip(*results)
    
    return np.array(T_arr), np.array(cluster_arr), np.array(spins_arr)

def main():
    
    N = 30
    melting_iterations = args.melting_iterations
    measuring_iterations = args.measuring_iterations

    temperature_array, graphing_cluster_array, spins_array  = run_sim(N, melting_iterations, measuring_iterations)
   
    for i in range(len(temperature_array)):
        x = np.arange(N)
        y = np.arange(N)
        X, Y = np.meshgrid(x,y)
        
        spins = spins_array[i]
        cluster = graphing_cluster_array[i]

        plt.figure()
        
        positive = spins == 1
        plt.scatter(X[positive], Y[positive], marker='s', facecolors='none', edgecolors='purple', label = r'spin $+1$')
        
        negative = spins == -1
        plt.scatter(X[negative], Y[negative], marker='+', color='green', label=r'spin $-1$')

        cluster_x = [c[1] for c in cluster]
        cluster_y = [c[0] for c in cluster]
        plt.scatter(cluster_x, cluster_y, marker='x', color='red', label=r'cluster')

        fig = plt.gcf()
        plt.title(f"2D Ising State at Temperature {temperature_array[i]}", pad = 20)
        fig.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, .92),
            ncol=3,
            frameon=False
        )   
        plt.subplots_adjust(top=0.85)
        plt.gca().set_aspect('equal')
        plt.savefig(f"2d_ising_state_{temperature_array[i]}.png")
        plt.show()
    '''
    fig, ax = plt.subplots(1,2)
    
    markersize=3
    labelpad=15
    plt.savefig("2d_ising_analytics.png")
    plt.show()
'''
if __name__ == "__main__":
    main()
