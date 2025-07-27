from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

# Load dataset
dataset = pd.read_csv('data.csv')
if dataset.empty:
    print("Dataset is empty. Please check the file.")
    exit()

# Number of clusters
k = 6
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(dataset)

# Get centroids and labels
gcenters = kmeans.cluster_centers_
labels = kmeans.labels_
colors = ['blue', 'red', 'green', 'black', 'yellow', 'brown']

# Plot clustered training data
for i in range(len(dataset)):
    plt.scatter(dataset.iloc[i, 0], dataset.iloc[i, 1], color=colors[labels[i]], s=10)

# Plot cluster centers
for i in range(k):
    plt.plot(gcenters[i, 0], gcenters[i, 1], 'kx', markersize=12)

# Test points
x_test = [[40.0, 67], [20.0, 61], [90.0, 90],
          [50.0, 54], [20.0, 80], [90.0, 60]]
predictions = kmeans.predict(x_test)
print("The predictions:")
print(predictions)

# Plot and annotate test points
for i, point in enumerate(x_test):
    plt.plot(point[0], point[1], 'm*', markersize=15)  # magenta star
    plt.text(point[0]+1.5, point[1]+1.5, f'Cluster {predictions[i]}', fontsize=9, color='magenta')
plt.title(f'No of clusters (k) = {k}')
plt.xlabel('Distance')
plt.ylabel('Location')
plt.grid(True)
plt.tight_layout()
plt.savefig('kmeans_output.png')