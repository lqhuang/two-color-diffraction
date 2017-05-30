import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,feature
from itertools import combinations

# create two circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
x3, y3 = 50, 20
r1, r2 = 16, 20
r3 = 10
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
mask_circle3 = (x - x3)**2 + (y - y3)**2 < r3**2
image = np.logical_or(mask_circle1, mask_circle2)
image = np.logical_or(image, mask_circle3)

distance = ndi.distance_transform_edt(image)  # distance transform
local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                    labels=image)
centorids = feature.peak_local_max(distance, indices=True, footprint=np.ones((3, 3)),
                                   labels=image)
print(centorids)
center_max_intensity = np.unravel_index(np.argmax(distance), distance.shape)

y, x = centorids[:, 0], centorids[:, 1]

markers = ndi.label(local_maxi)[0]  # mark initial points for watershed
labels = morphology.watershed(-distance, markers, mask=image)  # Watershed depend on distance transform

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, origin='lower')
ax0.set_title("Original")

ax1.imshow(distance, cmap=plt.cm.jet, origin='lower')
ax1.set_title("Distance")

ax2.imshow(markers, cmap=plt.cm.spectral, origin='lower')
ax2.plot(x, y, 'ro', fillstyle='none')

print(center_max_intensity)
slope_to_source = center_max_intensity[0] / center_max_intensity[1]  # slope = y / x

for (p1, p2) in combinations(centorids, 2):
    x_grid = np.arange(0, 80)
    center = (p1 + p2) / 2
    dist = np.sqrt( np.sum( (p1 - p2) ** 2 ) )
    ax2.text(center[1], center[0], '{:.2f}'.format(dist), color='white')
    ax2.plot((p1[1], p2[1]), (p1[0], p2[0]), 'b')
    ax2.plot(x_grid, slope_to_source * x_grid, '-')
    # ax2.axis([0, 80, 0, 80])
ax2.set_title("Markers")

ax3.imshow(labels, cmap=plt.cm.spectral, origin='lower')
ax3.set_title("Segmented")

# for ax in axes:
    # ax.axis('off')

fig.tight_layout()
plt.show()