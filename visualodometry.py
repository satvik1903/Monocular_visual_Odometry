import cv2 as cv
import numpy as np
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

        return good
    
    def process_frame(self,i):
        img=self.load_image(i)
        kp, ds=self.detect_features(img)
        if self.prev_ds is None:
            self.prev_ds=ds
            self.prev_kp=kp
            return

        good=self.feature_matching(ds,kp)
        self.prev_img = self.load_image(i - 1) 
        img1=cv.drawMatches(img,self.prev_kp,self.prev_img,kp,good,None)
        cv.imshow('drawn',img1)
        cv.waitKey(3000)
        print(len(good))
        
        #window slide
        self.prev_kp=kp
        self.prev_ds=ds


vo = VisualOdometry()
for i in range(10):
    vo.process_frame(i)

