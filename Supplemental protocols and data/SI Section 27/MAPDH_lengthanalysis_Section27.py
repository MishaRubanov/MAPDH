import matplotlib.pyplot as plt
import math
import glob
from skimage import io
from skimage.filters import threshold_otsu
from skimage.feature import corner_harris, corner_peaks
import csv

# Setting up csv file to compile swelling data
with open('singlegel_measurements.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(["file name", "length [um]"])
    
    # Processing each image in folder
    imagelist = []
    for filename in glob.glob('singlegel_1/*.jpg'):
        image = io.imread(filename)
        imagelist.append(image)
        fig, ax = plt.subplots()
        
        # Binary mask for the gels
        thresh = threshold_otsu(image)
        binary = image > thresh
        ax.imshow(binary, cmap=plt.cm.gray)
        
        
        coords = corner_peaks(corner_harris(binary), min_distance=10, threshold_rel=0.0)
        new_coords = [(y, x) for y, x in coords if x in range(0,90) and y in range (0,40)] 
        new_x = new_coords[1]
        new_y = new_coords[0]
        
#         x = coords[:, 1]
#         y = coords[:, 0]
        a = max(new_coords, key=lambda new_x: int(new_x[1]))
        b = min(new_coords, key=lambda new_x: int(new_x[1]))
        d = math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)
            
        # Print filename, coordinates of the references points, and distances
        print(filename)
        print(a[1], ",", a[0])
        print(b[1], ",", b[0])
        print(d)
        
        # Compile data into csv file
        csv_output.writerow([filename, d])
        
        # Show plotting and save
        plt.xlabel('x', fontsize=15, weight='normal')
        plt.ylabel('y', fontsize=15, weight='normal')
        ax.plot(new_coords[1], new_coords[0], color='red', marker='o',linestyle='None', markersize=6)
        ax.plot(a[1], a[0], color='cyan', marker='o',linestyle='None', markersize=10)
        ax.plot(b[1], b[0], color='cyan', marker='o',linestyle='None', markersize=10)
        ax.text(a[1], a[0], str((a[1], a[0])),fontsize=0,color = 'cyan')
        ax.text(b[1], b[0], str((b[1], b[0])),fontsize=0,color = 'cyan')
        ax.text((a[1]+a[0])/3,(b[0]+b[1])/1.5, str(d*2.03),fontsize=12,color = 'cyan')
        ax.plot(new_x, new_y, linestyle = '-.',  linewidth = 2, color = 'cyan')
        plt.savefig(filename + 'binary.svg')
        plt.show()