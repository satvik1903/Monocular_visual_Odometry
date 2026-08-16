#Lets start one by one first loading the calib file and useing its content

import numpy as np
import matplotlib.pyplot as plt   # put this at the top with your other imports

#Loading the calibration dataset 
with open('Dataset_Kitti/dataset_odometry_calib/sequences/00/calib.txt','r') as f:
    content=f.readline()
content_list=content.split()
filtered_list=content_list[1:]
array=np.asarray(filtered_list,dtype=float)
print(array)
array=array.reshape(3,4)
K=np.delete(array,3,axis=1)
print(f"reshaped:\n {K}")
#findint out focal lenght and center of image 
fx=K[0,0]
print(f"fx is: {fx}")
fy=K[1,1]
print(f"fy is: {fy}")
cx=K[0,2]
print(f"cx is: {cx}")
cy=K[1,2]
print(f"cy is: {cy}")
gtp=[]
#Loading the ground truth poses(understadn this again at home )
with open("Dataset_Kitti/dataset_odometry_poses/poses/00.txt") as f:
    for line in f:
        if line.strip():
            row=np.fromstring(line,sep=' ')
            array=row.reshape(3,4)
            trajectory=array[:,3]
            gtp.append(trajectory)
gtp=np.array(gtp)
print(gtp)
print(gtp.shape)

# gtp is your (4541, 3) trajectory array

xs = gtp[:, 0]    # all x values (column 0)
zs = gtp[:, 2]    # all z values (column 2) -- skipping y (column 1), the vertical

plt.figure(figsize=(8, 8))        # make a square canvas
plt.plot(xs, zs, color='blue', label='Ground Truth')  # draw the path
plt.xlabel('x (meters)')          # horizontal axis label
plt.ylabel('z (meters)')          # vertical axis label
plt.title('KITTI Sequence 00 — Ground Truth Trajectory (top-down)')
plt.axis('equal')                 # <-- the important one, see below
plt.legend()                      # show the label box
plt.grid(True)                    # gridlines, easier to read
plt.show()                        # render the window
            