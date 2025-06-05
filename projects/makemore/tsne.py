import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist["data"], mnist["target"]

tsne = TSNE()

X_embedded = tsne.fit_transform(X)

# Example assuming y contains integer class labels
classes = np.unique(y)
colors = plt.get_cmap('tab10', len(classes))

for i, class_label in enumerate(classes):
    plt.scatter(
        X_embedded[y == class_label, 0],
        X_embedded[y == class_label, 1],
        color=colors(i),
        label=str(class_label),
        edgecolors='k',  # optional: black edge for better visibility
        s=30             # optional: size of points
    )

plt.legend(title="Classes")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Scatter Plot with Labels")
plt.show()


tsne3d = TSNE(n_components=3)
X_embedded3d = tsne3d.fit_transform(X)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
for i, class_label in enumerate(classes):
    data = X_embedded3d[y==class_label]
    ax.scatter(
        data[:,0],
        data[:,1],
        data[:,2],
        color=colors(i),
        label=str(class_label),
        s=30, # type: ignore
        edgecolors='k'
    )
plt.legend()
plt.show()



