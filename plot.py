import matplotlib.pyplot as plt
from main import q, loss_values

epochs = list(range(0, q, 10))

plt.plot(epochs, loss_values)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()