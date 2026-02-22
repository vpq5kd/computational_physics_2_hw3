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
    
    
    for i in range(1000):
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

        if np.random.rand() < np.exp()
