import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 3, 5, 7])
y = np.array([1, 2, 2, 4])


weight_serie = lambda i: (i+2)/(2*(i+1))
# weight_serie = lambda i: 1/(i+1)
weighs = np.array([weight_serie(i) for i in range(len(x))])


p0 = np.polynomial.Polynomial.fit(x, y, 2)
p1 = np.polynomial.Polynomial.fit(x, y, 2, w=weighs)


x_min, x_max = x.min(), x.max()
margin = 0.1 * (x_max - x_min)
x_axis = np.linspace(x_min - margin, x_max + margin, 100)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].set_title("Weights")
axs[0].plot(weighs, 'o')
axs[0].grid()

axs[1].set_title("Fitting")
axs[1].plot(x, y, 'o')
axs[1].plot(x_axis, p0(x_axis), label='fit')
axs[1].plot(x_axis, p1(x_axis), label='weighted fit')
axs[1].legend()
axs[1].grid()

plt.show()
