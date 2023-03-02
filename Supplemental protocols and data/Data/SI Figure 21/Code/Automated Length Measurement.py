'''This is a code for automated hydrogel length measurement'''

# Import necessary packages
import matplotlib.pyplot as plt
import math
import glob
from skimage import io
from skimage.filters import threshold_otsu
from skimage.feature import corner_harris, corner_peaks
import csv

# Setting up csv file to compile swelling data
with open('gelswellingdata.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(["file name", "length [um]"])
    
    # Processing each image in folder
    imagelist = []
    for filename in glob.glob('Figure S21/*.jpg'):
        image = io.imread(filename)
        imagelist.append(image)
        fig, ax = plt.subplots()
        
        # Binary mask for the gels
        thresh = threshold_otsu(image)
        binary = image > thresh
        ax.imshow(binary, cmap=plt.cm.gray)
        
        coords = corner_peaks(corner_harris(binary), min_distance=10, threshold_rel=0)
        x = coords[:, 1]
        y = coords[:, 0]
#         new_coords = [(y, x) for y, x in coords if y in range (20, 100)] 
#         new_x = new_coords[1]
#         new_y = new_coords[0]
    
        a = max(coords, key=lambda x: x[1])
        b = min(coords, key=lambda x: x[1])
        d = math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)
            
        # Print filename, coordinates of the references points, and distances
        print(filename)
        print(a[1], ",", a[0])
        print(b[1], ",", b[0])
        print(d)
        
        # Compile data into csv file
        csv_output.writerow([filename, d])
        
        # Show plotting
        ax.axis((0, 200, 0, 160))
        ax.plot(x, y, color='red', marker='o',linestyle='None', markersize=2)
        ax.plot(a[1], a[0], color='cyan', marker='o',linestyle='None', markersize=6)
        ax.plot(b[1], b[0], color='cyan', marker='o',linestyle='None', markersize=6)
        ax.text(a[1], a[0], str((a[1], a[0])),fontsize=20,color = 'cyan')
        ax.text(b[1], b[0], str((b[1], b[0])),fontsize=20,color = 'cyan')
        plt.show()




