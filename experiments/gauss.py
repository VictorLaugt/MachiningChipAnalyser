import matplotlib.pyplot as plt
import numpy as np

x_arr = np.linspace(-5, 5, 1000)

def gaussian(x, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

def laplacian_of_gaussian(x, sigma):
    return -1/(np.pi * sigma**4) * (1 - (x**2 / (2 * sigma**2))) * np.exp(-x**2 / (2 * sigma**2))


fig, axis = plt.subplots(2, 1, figsize=(10, 10))

for sigma in [1.4]:
    axis[0].plot(x_arr, [gaussian(x, sigma) for x in x_arr], label=f'sigma={sigma}')
    axis[1].plot(x_arr, [laplacian_of_gaussian(x, sigma) for x in x_arr], label=f'sigma={sigma}')

    axis[0].set_title(f'Gaussian')
    axis[1].set_title(f'Laplacian of Gaussian')

    axis[0].legend()
    axis[1].legend()
    axis[0].grid()
    axis[1].grid()



axis[1].plot(x_arr, [500*laplacian_of_gaussian(x, 1.4) for x in x_arr], linestyle='--', color='black')

plt.xticks(np.arange(x_arr[0], x_arr[-1], 1))
plt.show()


