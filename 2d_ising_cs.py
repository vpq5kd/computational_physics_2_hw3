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
        return delta_e

    return 0

def metropolis(N,T,spins,melting_iterations,measuring_iterations):

    temp_energy_array = []
    
    for i in range(melting_iterations):
        for i in range(N**2):
            metropolis_logic(N, T, spins)
    
    for i in range(measuring_iterations):
        for i in range(N**2):
            metropolis_logic(N, T, spins)
        temp_energy_array.append(hamiltonian(spins))

    return np.mean(temp_energy_array), np.mean(np.array(temp_energy_array)**2)

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


def wolff_cluster(N, T, spins, melting_iterations, measuring_iterations):
    
    temp_energy_array = []
   
    for i in range(melting_iterations):
        wolff_cluster_logic(N, T, spins)    
    
    for i in range(measuring_iterations):
        wolff_cluster_logic(N, T, spins)
        temp_energy_array.append(hamiltonian(spins))

    return np.mean(temp_energy_array), np.mean(np.array(temp_energy_array)**2)

def temperature_logic(args):
    
    N, T, melting_iterations, measuring_iterations = args

    #spins_metropolis = np.random.choice([-1,1], size=(N,N))
    #spins_wolff = np.random.choice([-1,1], size=(N,N))

    spins_metropolis = np.ones((N,N))
    spins_wolff = np.ones((N,N))

    wolff_energy, wolff_energy_squared = wolff_cluster(N, T, spins_wolff, melting_iterations, measuring_iterations)
    metropolis_energy, metropolis_energy_squared  = metropolis(N, T, spins_metropolis, melting_iterations, measuring_iterations)

    return T, wolff_energy, wolff_energy_squared, metropolis_energy, metropolis_energy_squared

def run_sim(N, melting_iterations, measuring_iterations):
    
    temperature_array = np.linspace(0.1, 5.1, 100)[::-1]
    
    args = [(N, T, melting_iterations, measuring_iterations) for T in temperature_array]
   
    print(f"Beginning simulation with {os.cpu_count()} cores.")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(temperature_logic, args), total=len(args)))

    T_arr, wolff_e_arr, wolff_e_2_arr, metro_e_arr, metro_e_2_arr = zip(*results)
    return np.array(T_arr), np.array(wolff_e_arr), np.array(wolff_e_2_arr), np.array(metro_e_arr), np.array(metro_e_2_arr)

def main():
    
    N = 30
    melting_iterations = args.melting_iterations
    measuring_iterations = args.measuring_iterations

    temperature_array, wolff_energy_array, wolff_energy_array_squared, metropolis_energy_array, metropolis_energy_array_squared  = run_sim(N, melting_iterations, measuring_iterations)
    
    average_wolff_energy_array = wolff_energy_array/N**2
    average_wolff_energy_array_squared = wolff_energy_array_squared/N**2
    average_metro_energy_array = metropolis_energy_array/N**2
    average_metro_energy_array_squared = metropolis_energy_array_squared/N**2
    
    wolff_heat_cap_array = (1/temperature_array**2) * (wolff_energy_array_squared - wolff_energy_array**2) / N**2

    metro_heat_cap_array = (1/temperature_array**2) * (metropolis_energy_array_squared - metropolis_energy_array**2) / N**2
    
    fig, ax = plt.subplots(1,2)
    
    markersize=3
    labelpad=15
    color='green'

    ax[0].plot(temperature_array, average_metro_energy_array,label='metropolis',color='red',marker='o',markersize=markersize,markerfacecolor='none',linewidth=1)
    ax[0].plot(temperature_array, average_wolff_energy_array,'o',markerfacecolor='none', label='wolff',markersize=markersize, color=color)
    ax[0].set_xlabel("T")
    ax[0].set_ylabel(r"$\frac{E}{N}$", rotation = 0, labelpad=labelpad)
    ax[1].plot(temperature_array, metro_heat_cap_array,color='red',marker='o',markersize=markersize,markerfacecolor='none',linewidth=1)
    ax[1].plot(temperature_array, wolff_heat_cap_array, 'o', markerfacecolor='none', markersize=markersize, color = color)
    ax[1].set_xlabel("T")
    ax[1].set_ylabel(r"$\frac{C}{N}$", rotation = 0, labelpad=labelpad)
    
    fig.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig("2d_ising_plots_cs.png")
    plt.show()

if __name__ == "__main__":
    main()
