import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
class VisualOdometry():

    def __init__(self):
        self.K=self.calibration()
        self.orb=cv.ORB_create()
        self.matcher=cv.BFMatcher(cv.NORM_HAMMING)
        self.prev_ds=None
        self.prev_kp=None
        self.cur_pose = np.eye(4)
        self.trajectory = []
        self.prev_img=None
        #will add the rest of the things here are there respectie function are written

    #function for loading the images one by one to save on space 
    def load_image(self,index):
        path=f"Dataset_Kitti/dataset_odometry_gray/sequences/00/image_0/{index:06d}.png"
        img=cv.imread(path,0)
        if img is None:
            raise FileNotFoundError(f"Image loading issue")
        return img

    #function to get the K(intrinsic of the camera at PO)
    def calibration(self):
        calib_path=f"Dataset_Kitti/dataset_odometry_calib/sequences/00/calib.txt"
        with open(calib_path,'r') as f:
            calib_data=f.readline()
        data_list=calib_data.split()
        data_updated=data_list[1:]
        data_array=np.asarray(data_updated,dtype=float)
        K_matrix=data_array.reshape(3,4)
        K=np.delete(K_matrix,3,axis=1)
        return K

    #the ORB(Orient FAST and rotate Brief) feature detection and matching function 
    def detect_features(self,img):
        kp,ds=self.orb.detectAndCompute(img,None)
        return kp, ds

    def feature_matching(self,ds,kp):
        matches=self.matcher.knnMatch(self.prev_ds,ds,k=2)
        good=[]
        for m,n in matches:
            if m.distance<0.75*n.distance:
                good.append(m)
        prev_pts=np.array([self.prev_kp[m.queryIdx].pt for m in good])
        cur_pts=np.array([kp[m.trainIdx].pt for m in good])
        return prev_pts, cur_pts

    def estimate_motion(self,prev_pts,cur_pts):
        #Find E(Essential Matrix from the matched point)
        E, mask=cv.findEssentialMat(cur_pts,prev_pts,self.K,method=cv.RANSAC,prob=0.999,threshold=1.0)

        #recover the r and t now 
        inliercnt,R,t,mask=cv.recoverPose(E,cur_pts,prev_pts,self.K,mask=mask)
        return R, t

    def process_frame(self,i):
        img=self.load_image(i)
        kp, ds=self.detect_features(img)
        if self.prev_ds is None:
            self.prev_ds=ds
            self.prev_kp=kp
            return

        prev_pts,cur_pts=self.feature_matching(ds,kp)#we are catching the points here that has died in the return of
        #of the feture matching function

        R,t=self.estimate_motion(prev_pts,cur_pts)

        T=np.eye(4)
        T[:3,:3]=R
        T[:3,3]=t.ravel()

        self.cur_pose=self.cur_pose@T
        self.trajectory.append(self.cur_pose[:3,3])
        


        #window slide
        self.prev_kp=kp
        self.prev_ds=ds


vo = VisualOdometry()
for i in range(4000):
    vo.process_frame(i)

traj = np.array(vo.trajectory)
plt.plot(traj[:, 0], traj[:, 2])   # x vs z, top-down
plt.axis('equal')
plt.show()  

