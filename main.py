from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#img = Image.open(r"COLOR_GEN\image.jpg")
img = Image.open(r"image.jpg")
rgb_img = img.convert('RGB')
rgb_img.show()

#downsize pixels
width,height= rgb_img.size
print(width*height)

rgb_img.thumbnail((244,244), Image.Resampling.LANCZOS)
rgb_img.show()

#convert to array
rgb_array=np.array(rgb_img)
print(rgb_array)
#flatten
pixels=rgb_array.reshape(-1,3)
#clustering
num_colors = 5

kmeans = KMeans(n_clusters=num_colors, random_state=42)
kmeans.fit(pixels)
colors = kmeans.cluster_centers_.astype(int)


# Count pixels in each cluster
labels = kmeans.labels_
counts = np.bincount(labels)

# Calculate percentages
percentages = counts / len(labels) * 100

# Sort colors by percentage
order = np.argsort(percentages)[::-1]

colors = colors[order]
percentages = percentages[order]

# -------------------------------
# Display Results
# -------------------------------
print("Dominant Colors:\n")

for i, (color, percent) in enumerate(zip(colors, percentages), start=1):
    r, g, b = color
    hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)

    print(f"Color {i}")
    print(f"RGB : ({r}, {g}, {b})")
    print(f"HEX : {hex_code}")
    print(f"Percentage : {percent:.2f}%")
    print("-" * 30)

# -------------------------------
# Create Palette Image
# -------------------------------
palette = np.zeros((100, 500, 3), dtype=np.uint8)

start = 0

for color in colors:
    end = start + 100
    palette[:, start:end] = color
    start = end

plt.figure(figsize=(8,2))
plt.imshow(palette)
plt.axis("off")
plt.title("Dominant Color Palette")
plt.show()


# ==========================================
# DBSCAN Color Clustering
# ==========================================

print("\n==============================")
print("DBSCAN Dominant Colors")
print("==============================")

dbscan = DBSCAN(eps=3, min_samples=30)

db_labels = dbscan.fit_predict(pixels)

# Remove noise (-1 labels)
mask = db_labels != -1

db_pixels = pixels[mask]
db_labels = db_labels[mask]

unique_labels = np.unique(db_labels)

db_colors = []
db_percentages = []

for label in unique_labels:

    cluster_pixels = db_pixels[db_labels == label]

    mean_color = cluster_pixels.mean(axis=0).astype(int)

    db_colors.append(mean_color)

    db_percentages.append(len(cluster_pixels))

db_colors = np.array(db_colors)
db_percentages = np.array(db_percentages)

# Convert counts to percentages
db_percentages = db_percentages / db_percentages.sum() * 100

# Sort by percentage
order = np.argsort(db_percentages)[::-1]

db_colors = db_colors[order]
db_percentages = db_percentages[order]

# Print colors
for i, (color, percent) in enumerate(zip(db_colors, db_percentages), start=1):

    r, g, b = color
    hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)

    print(f"Color {i}")
    print(f"RGB : ({r}, {g}, {b})")
    print(f"HEX : {hex_code}")
    print(f"Percentage : {percent:.2f}%")
    print("-" * 30)

# ------------------------------------------
# Display DBSCAN Palette
# ------------------------------------------

palette = np.zeros((100, 500, 3), dtype=np.uint8)

start = 0

segment_width = 500 // len(db_colors)

for color in db_colors:

    end = start + segment_width
    palette[:, start:end] = color
    start = end

plt.figure(figsize=(8,2))
plt.imshow(palette)
plt.axis("off")
plt.title("DBSCAN Color Palette")
plt.show()