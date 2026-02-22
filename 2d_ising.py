import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

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

def metropolis(N,T,spins,iterations):

    temp_energy_array = []
    initial_energy = hamiltonian(spins)
    temp_energy_array.append(initial_energy)
    for i in range(iterations):
        previous_energy = temp_energy_array[-1]

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
            temp_energy_array.append(previous_energy+delta_e)
            continue

        temp_energy_array.append(previous_energy)

    return np.array(temp_energy_array)

def wolff_cluster(N, T, spins, iterations):
    
    temp_energy_array = []
    initial_energy = hamiltonian(spins)
    temp_energy_array.append(initial_energy)
    for i in range(iterations):
        previous_energy = temp_energy_array[-1]

        random_site_x = rng.integers(N)
        random_site_y = rng.integers(N)
        
        random_site = (random_site_x, random_site_y)
        cluster = set()
        cluster.append(random_site)
        f_old = set()
        f_old.append(random_site)

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
                            f_new.append(neighbor)
                            cluster.append(neighbor)
            f_old = f_new

        for spin in cluster:
            x = spin[1]
            y = spin[0]

            spins[y,x]*=-1
        
        temp_energy_array.append(hamiltonian(spins))

        return np.mean(temp_energy_array)

def run_sim(N,iterations):
    
    spins_metropolis = np.random.choice([-1,1], size=(N,N))
    spins_wolff = np.random.choice([-1,1], size=(N,N))
    temperature_array = np.linspace(0.1, 5.1, 100)[::-1]
    metropolis_energy_array = []
    wolff_energy_array = []

    for T in temperature_array:
        temperature_wolff_mean_energy = wolff_cluster(N, T, spins_wolff, iterations)
        temperature_metropolis_mean_energy = metropolis(N, T, spins_metropolis, iterations)
        
        wolff_energy_array.append(temperature_wolff_mean_energy)
        metropolis_energy_array.append(temperature_metropolis_mean_energy)

    return np.array(temperature_array), np.array(wolff_energy_array), np.array(metropolis_energy_array)


def main():
    N = 30
    iterations = 1000
    temperature_array, wolff_energy_array, metropolis_energy_array = run_sim(N, iterations)
    plt.figure()
    plt.plot(temperature_array, metropolis_energy_array/N**2)
    plt.plot(temperature_array, wolff_energy_array/N**2)
    plt.show()

main()
