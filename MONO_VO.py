import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class VisualOdometry():

    def __init__(self):
        self.K=self.calibration()
        self.orb=cv.ORB_create(5000)
        self.matcher=cv.BFMatcher(cv.NORM_HAMMING)
        self.prev_ds=None
        self.prev_kp=None
        self.cur_pose = np.eye(4)
        self.trajectory = []
        self.prev_img=None
        self.gt_poses=self.load_ground_truth()

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
        E, mask=cv.findEssentialMat(cur_pts,prev_pts,self.K,method=cv.RANSAC,prob=0.999,threshold=0.4)

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
        scale = np.linalg.norm(self.gt_poses[i] - self.gt_poses[i-1])  # true distance moved
        t = t * scale    

        T=np.eye(4)
        T[:3,:3]=R
        T[:3,3]=t.ravel()

        self.cur_pose=self.cur_pose@T
        self.trajectory.append(self.cur_pose[:3,3])
        
        #window slide
        self.prev_kp=kp
        self.prev_ds=ds


    def load_ground_truth(self):
        gt_path="Dataset_Kitti/dataset_odometry_poses/poses/00.txt"
        gt=[]
        with open(gt_path,"r") as f:
            for line in f:
                if line.strip():
                    row=np.fromstring(line,sep=" ")
                    mat=row.reshape(3,4)
                    traje=mat[:,3]
                    gt.append(traje)
        return np.array(gt)


def align(est,gt):
        mean_est=est.mean(axis=0)
        mean_gt=gt.mean(axis=0)
        est_c=est-mean_est
        gt_c=gt-mean_gt
        H=est_c.T @gt_c
        U,S,Vt=np.linalg.svd(H) #U=orientation of estimated data Vt=Orientation of ground truth S=how strongly those direction correspond
        R=Vt.T@U.T
        if np.linalg.det(R) < 0: #If R is anywere negative that means there is a reflection which we do not want
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mean_gt - R @ mean_est
        return (R @ est.T).T + t

def compute_rpe(est, gt):
    # relative (frame-to-frame) motions
    est_rel = est[1:] - est[:-1]     # each row: motion from frame i to i+1, estimated
    gt_rel  = gt[1:]  - gt[:-1]      # same, ground truth
    diff    = est_rel - gt_rel       # how wrong each relative step is
    errors  = np.linalg.norm(diff, axis=1)   # distance error per step
    rpe = np.sqrt(np.mean(errors**2))        # RMS
    return rpe

rpe = compute_rpe(est, gt)
print(f"RPE (RMSE): {rpe:.3f} meters")
        
vo = VisualOdometry()
for i in range(1941): #total is 4541
    vo.process_frame(i)

est=np.array(vo.trajectory)
gt=vo.gt_poses[1:len(est)+1]
print(est.shape,gt.shape)

traj = np.array(vo.trajectory)
plt.plot(traj[:, 0], traj[:, 2], label='Estimated VO')
plt.plot(vo.gt_poses[:, 0], vo.gt_poses[:, 2], label='Ground Truth')
plt.axis('equal')
plt.legend()
plt.show()

plt.plot(traj[:200, 0], traj[:200, 2], label='Est')
plt.plot(vo.gt_poses[:200, 0], vo.gt_poses[:200, 2], label='GT')
plt.axis('equal'); plt.legend(); plt.show()

est_aligned = align(est, gt)
# ... your ATE computation here ...
est_aligned = align(est, gt)

diff = est_aligned - gt                 # per-frame difference, (N,3)
errors = np.linalg.norm(diff, axis=1)   # per-frame distance, (N,)
ate = np.sqrt(np.mean(errors**2))       # RMS of the distances → one number

print(f"ATE (RMSE): {ate:.2f} meters")
print(f"ATE (RMSE): {ate:.2f} meters")

plt.plot(est_aligned[:, 0], est_aligned[:, 2], label='Estimated (aligned)')
plt.plot(gt[:, 0], gt[:, 2], label='Ground Truth')
plt.axis('equal'); plt.legend(); plt.show()