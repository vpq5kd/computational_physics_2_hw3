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

def metropolis(N):
    spins = np.random.choice([-1,1], size=(N,N))
    
    temp_array = np.linspace(0.1, 5.1, 100)[::-1]
    energy_array = []

    for T in temp_array:
        temp_energy_array = []
        initial_energy = hamiltonian(spins)
        temp_energy_array.append(initial_energy)
        for i in range(1000000):
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
        print(T)
        energy_array.append(np.mean(temp_energy_array))    

    return np.array(temp_array), np.array(energy_array)

def main():
    N = 30
    metropolis_temp_array, metropolis_energy_array = metropolis(30)
    plt.figure()
    plt.plot(metropolis_temp_array, metropolis_energy_array/N**2)
    plt.show()

main()
