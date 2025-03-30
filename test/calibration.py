import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

class FisheyeCameraCalibration:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
    
    def find_checkerboard_corners(self, images_folder):
        images = glob.glob(os.path.join(images_folder, '*.jpg')) + \
                 glob.glob(os.path.join(images_folder, '*.png')) + \
                 glob.glob(os.path.join(images_folder, '*.jpeg'))
        
        processed_images = []
        valid_images = []
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"Could not read image: {fname}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Try different methods of corner detection
            methods = [
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                cv2.CALIB_CB_ADAPTIVE_THRESH,
                cv2.CALIB_CB_NORMALIZE_IMAGE,
                0
            ]
            
            ret = False
            for method in methods:
                ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, method)
                if ret:
                    break
            
            if ret:
                self.objpoints.append(self.objp)
                
                # More aggressive corner refinement
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2.reshape(1, -1, 2).astype(np.float64))
                
                # Draw and save processed images
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, self.checkerboard_size, corners2, ret)
                processed_images.append(img_with_corners)
                valid_images.append((fname, img))
            else:
                print(f"Could not find checkerboard corners in {fname}")
        
        return processed_images, valid_images
    
    def calibrate_camera(self, image_size):
        if len(self.objpoints) < 10:
            raise ValueError(f"Not enough calibration images. Found {len(self.objpoints)}, need at least 10.")
        
        # Try multiple calibration strategies
        flags_list = [
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            0
        ]
        
        for flags in flags_list:
            try:
                K = np.zeros((3, 3))
                D = np.zeros((4, 1))
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(self.objpoints))]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(self.objpoints))]
                
                rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    self.objpoints, 
                    self.imgpoints, 
                    image_size, 
                    K, 
                    D, 
                    rvecs, 
                    tvecs, 
                    flags, 
                    self.criteria
                )
                
                print(f"Calibration successful with flags {flags}. RMS error: {rms}")
                return K, D, rvecs, tvecs
            
            except cv2.error as e:
                print(f"Calibration failed with flags {flags}: {e}")
        
        raise ValueError("All calibration attempts failed")
    
    def undistort_image(self, img, K, D, balance=0.0):
        """
        Undistort an image with optional balance parameter to control field of view
        
        :param img: Input image
        :param K: Camera matrix
        :param D: Distortion coefficients
        :param balance: Balance parameter to control FOV (0.0 to 1.0)
        :return: Undistorted image
        """
        dim1 = img.shape[:2][::-1]
        
        # Compute new camera matrix
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, dim1, np.eye(3), balance=balance
        )
        
        # Create map for undistortion
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2
        )
        
        # Remap the image
        undistorted_img = cv2.remap(
            img, map1, map2, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return undistorted_img

def debug_calibration(images_folder):
    # Create calibration object
    calibrator = FisheyeCameraCalibration(
        checkerboard_size=(9, 6),  # Adjust based on your checkerboard
        square_size=0.025  # Size of each square in meters
    )
    
    # Find and process checkerboard corners
    processed_images, valid_images = calibrator.find_checkerboard_corners(images_folder)
    
    # Display processed images with corners
    for img in processed_images:
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(200)
    cv2.destroyAllWindows()
    
    # Calibrate camera
    if valid_images:
        image_size = valid_images[0][1].shape[:2][::-1]
        
        try:
            K, D, rvecs, tvecs = calibrator.calibrate_camera(image_size)
            # Print detailed calibration results
            print("\nCamera Matrix:\n", K)
            print("\nDistortion Coefficients:\n", D)
            
            # Visualize undistortion with different balance parameters
            plt.figure(figsize=(15, 5))
            
            balance_values = [0.0, 0.5, 1.0]
            for i, balance in enumerate(balance_values):
                # Use first valid image for demonstration
                sample_image = valid_images[0][1]
                
                # Undistort with different balance values
                undistorted = calibrator.undistort_image(sample_image, K, D, balance)
                
                plt.subplot(1, 3, i+1)
                plt.title(f'Balance = {balance}')
                plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            return K, D
        
        except Exception as e:
            print(f"Calibration failed: {e}")
    
    return None, None

# Run the debug script
if __name__ == '__main__':
    images_folder = '../examples/cache/image'
    debug_calibration(images_folder)