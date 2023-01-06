import create_environment
import biped
import matplotlib.pyplot as plt

region_centers = [[0, 0], [2, 0], [4, 0], [6, 0], [6, 3], [4, 3],
                  [2, 3], [0, 3], [0, 6], [2, 6], [4, 6], [6, 6], [6, 9]]
for center in region_centers:
    create_environment.createSquare(center, 0.5)
plt.axis('scaled')
plt.show()
