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

def metropolis_logic(N, T, spins):

    random_site_x = rng.integers(N)
    random_site_y = rng.integers(N)

    right = spins[random_site_y, (random_site_x + 1) % N]
    down = spins[(random_site_y + 1) % N, random_site_x]
    left = spins[random_site_y, (random_site_x - 1)%N]
    up = spins[(random_site_y - 1) % N, random_site_x]
    neighbors = [up, down, right, left]

    h_mol = 0
    for neighbor in neighbors:
        h_mol += neighbor

    sigma_i = spins[random_site_y, random_site_x]
    delta_e = 2 * sigma_i * h_mol

    B = 1/T
    if np.random.rand() < np.exp(-B*delta_e):
        spins[random_site_y, random_site_x] *= -1
        return True

    return False

def metropolis(N,T,spins,melting_iterations,measuring_iterations):

    acceptances = 0
    count = 0
    for i in range(melting_iterations):
        for i in range(N**2):
            acceptance = metropolis_logic(N, T, spins)
            if acceptance:
                acceptances += 1
            count += 1

    for i in range(measuring_iterations):
        for i in range(N**2):
            acceptance = metropolis_logic(N, T, spins)
            if acceptance:
                acceptances+=1
            count+=1

    acceptance_rate = acceptances/count
    return acceptance_rate

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

    return len(cluster)

def wolff_cluster(N, T, spins, melting_iterations, measuring_iterations):
    
    temp_cluster_size_array = []
   
    for i in range(melting_iterations):
        temp_cluster_size_array.append(wolff_cluster_logic(N, T, spins))
    
    for i in range(measuring_iterations):
        temp_cluster_size_array.append(wolff_cluster_logic(N, T, spins))

    return np.mean(np.array(temp_cluster_size_array))

def temperature_logic(args):
    
    N, T, melting_iterations, measuring_iterations = args

    #spins_metropolis = np.random.choice([-1,1], size=(N,N))
    #spins_wolff = np.random.choice([-1,1], size=(N,N))

    spins_metropolis = np.ones((N,N))
    spins_wolff = np.ones((N,N))

    cluster_size = wolff_cluster(N, T, spins_wolff, melting_iterations, measuring_iterations)
    acceptance_rate = metropolis(N, T, spins_metropolis, melting_iterations, measuring_iterations)

    return T, cluster_size, acceptance_rate

def run_sim(N, melting_iterations, measuring_iterations):
    
    temperature_array = np.linspace(0.1, 5.1, 100)[::-1]
    
    args = [(N, T, melting_iterations, measuring_iterations) for T in temperature_array]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(temperature_logic, args), total=len(args)))

    T_arr, cluster_size_arr, acceptance_rate_arr = zip(*results)
    return np.array(T_arr), np.array(cluster_size_arr), np.array(acceptance_rate_arr)
def main():
    
    N = 30
    melting_iterations = args.melting_iterations
    measuring_iterations = args.measuring_iterations

    temperature_array, cluster_size_array, acceptance_rate_array = run_sim(N, melting_iterations, measuring_iterations)
    
    fig, ax = plt.subplots(1,2)
    
    markersize=3
    labelpad=15
    ax[0].plot(temperature_array, cluster_size_array, marker='o',markersize=markersize,color='purple',markerfacecolor='none',linewidth=0)
    ax[0].set_xlabel(r"$T$")
    ax[0].set_ylabel(r"Wolff Algorithm Cluster Size", labelpad=labelpad)
    ax[1].plot(temperature_array, acceptance_rate_array, marker='o',markersize=markersize,color='orange',markerfacecolor='none',linewidth=0)
    ax[1].set_xlabel(r"T")
    ax[1].set_ylabel(r"Metropolis Algorithm Acceptance Rate", labelpad=labelpad)
    fig.tight_layout()
    plt.savefig("2d_ising_analytics.png")
    plt.show()

if __name__ == "__main__":
    main()
