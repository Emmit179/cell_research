import numpy as np
from skimage import io
from skimage.measure import label, regionprops
from scipy.ndimage import rotate, binary_closing

# load file with skimage
ground_truth = io.imread('data/00_Cell1_2_Crop1_Back3_steve_ground_truth.tif')

ground_truth_flipped = np.swapaxes(ground_truth, 0, 2)

# closed_ground_truth = binary_closing(ground_truth_flipped)

labeled_volume = label(ground_truth_flipped)

pores = regionprops(labeled_volume)

def vertical(coords, centroid):
    cur = round(centroid[1])

    radius = False

    while coords[round(centroid[0])][cur][round(centroid[2])] == 0 and coords[round(centroid[0])][cur][round(centroid[2]) - 1] == 0 and coords[round(centroid[0])][cur][round(centroid[2]) + 1] == 0:
        cur += 1

        if (len(coords[round(centroid[0])]) <= cur):
            radius = True
            break

    other = round(centroid[1])

    while coords[round(centroid[0])][other][round(centroid[2])] == 0 and coords[round(centroid[0])][other][round(centroid[2]) - 1] == 0 and coords[round(centroid[0])][other][round(centroid[2]) + 1] == 0:
        other -= 1

        if (other < 0):
            return abs(cur - centroid[1]) * 2

    if radius:
        return abs(other - centroid[1]) * 2
    
    return abs(cur - other)

def horizontal(coords, centroid):
    cur = round(centroid[0])

    radius = False

    while coords[cur][round(centroid[1])][round(centroid[2])] == 0 and coords[cur][round(centroid[1])][round(centroid[2]) - 1] == 0 and coords[cur][round(centroid[1])][round(centroid[2]) + 1] == 0:
        cur += 1

        if (len(coords) <= cur):
            radius = True
            break

    other = round(centroid[0])

    while coords[other][round(centroid[1])][round(centroid[2])] == 0 and coords[other][round(centroid[1])][round(centroid[2]) - 1] == 0 and coords[other][round(centroid[1])][round(centroid[2]) + 1] == 0:
        other -= 1

        if (other < 0):
            return abs(cur - centroid[0]) * 2

    if radius:
        return abs(other - centroid[0]) * 2

    return abs(cur - other)
    

i = 1

threshold = 150

for pore in pores:
    if len(pore.coords) < threshold:
        continue
    # print(len(pore.coords))
    angles = np.arange(0, 90, 18)

    diameters = []

    # for each angle 10 deg 18 times
    for angle in angles:
        # Rotate the region by a specific angle (e.g., 10 degrees)
        rotated_region = rotate(labeled_volume == pore.label, angle=angle, axes=(0, 1), order=0, mode='nearest').astype(int)
        temp = regionprops(rotated_region)

        centroid = temp[0].centroid
        
        diameters.append(vertical(rotated_region, centroid))
        diameters.append(horizontal(rotated_region, centroid))

    # get mean diameter of all angles
    average_diameter = np.mean(diameters)

    print(f"Pore Centroid: {pore.centroid} Diameter: {average_diameter}")

    i += 1