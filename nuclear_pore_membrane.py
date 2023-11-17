import numpy as np
from skimage import io
from skimage.measure import label, regionprops
import sys

# load file with skimage
ground_truth = io.imread('data/00_Cell1_2_Crop1_Back3_steve_ground_truth.tif')

#function for rotating coords around centroid
def rotate(coords, centroid, angle):
    angle_rad = np.radians(angle)

    translated_coords = coords - centroid

    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])

    rotated_coords = np.dot(translated_coords, rotation_matrix.T)

    rotated_coords += centroid

    return rotated_coords

# divide into pores and use regionprops for centroid/labels
labeled_pores = label(ground_truth)

pore_props = regionprops(labeled_pores)

for pore in pore_props:
    coords = np.array(pore.coords)
    
    angles = np.arange(0, 180, 10)
    diameters = []

    # for each angle 10 deg 18 times
    for angle in angles:
        centroid = pore.centroid

        rotated_coords = np.array(rotate(coords, centroid, angle))
        
        # create x axis at angle then plug into original image for intensities along axis
        x_intensity = []
        for x_cord in rotated_coords[:, 0]:
            y_cord = centroid[1]
            z_cord = centroid[2]
            # print([int(x_cord), int(y_cord), int(z_cord)])
            x_intensity.append(ground_truth[int(x_cord), int(y_cord), int(z_cord)])
        
        #repeat for y axis
        y_intensity = []
        for y_cord in rotated_coords[:, 1]:
            x_cord = centroid[0]
            z_cord = centroid[2]
            y_intensity.append(ground_truth[int(x_cord), int(y_cord), int(z_cord)])

        # get maxima indices along axes (i think instead of min because alphas are inverse to example)
        x_maxima_indices = np.where(x_intensity == np.max(x_intensity))[0]
        y_maxima_indices = np.where(y_intensity == np.max(y_intensity))[0]

        # exit()

        # get difference of pixels and subtract one to get number of pixels between both maxes
        x_maxima_distance = np.abs(x_maxima_indices[1] - x_maxima_indices[0] - 1)
        y_maxima_distance = np.abs(y_maxima_indices[1] - y_maxima_indices[0] - 1)

        # append both to diameter
        diameters.append(x_maxima_distance)
        diameters.append(y_maxima_distance)

    # get mean diameter of all angles
    average_diameter = np.mean(diameters)

    print(f"Pore Number {pore.label} Diameter: {average_diameter}")