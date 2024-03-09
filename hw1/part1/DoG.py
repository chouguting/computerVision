import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        sigmaValues = [self.sigma**i for i in range(1,self.num_DoG_images_per_octave+1)]
        gaussian_images = []
        imageForGaussion = image.copy()

        for octives in range(self.num_octaves):
            gaussian_images.append([imageForGaussion])
            gaussian_images[-1].extend(cv2.GaussianBlur(imageForGaussion, (0,0), sigma) for sigma in sigmaValues)
            imageForGaussion = cv2.resize(gaussian_images[-1][-1], (0, 0), fx=0.5, fy=0.5 , interpolation=cv2.INTER_NEAREST) #resize the image for next octave (half the size of the last image in the octave)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for octives in range(self.num_octaves):
            dog_images.append([cv2.subtract(gaussian_images[octives][i+1], gaussian_images[octives][i]) for i in range(len(gaussian_images[octives])-1)])


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        scale = 1
        for octives in range(self.num_octaves):
            octiveKeyPoints = set()
            octivesDogImageArray = np.array(dog_images[octives])
            for dogIndex in range(1,octivesDogImageArray.shape[0]-1):
                currentDogImage = octivesDogImageArray[dogIndex]
                overThreshold = np.argwhere(abs(currentDogImage) > self.threshold)
                for(x,y) in overThreshold:
                    if dogIndex == 0 or dogIndex == octivesDogImageArray.shape[0]-1:
                        continue
                    if (currentDogImage[x,y] == np.max(octivesDogImageArray[dogIndex-1:dogIndex+2, x-1:x+2, y-1:y+2]) or currentDogImage[x,y] == np.min(octivesDogImageArray[dogIndex-1:dogIndex+2, x-1:x+2, y-1:y+2])):
                        octiveKeyPoints.add((x * scale,y * scale))
            scale *= 2
            keypoints.extend(octiveKeyPoints)


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
