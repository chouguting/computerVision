
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        imageForFilter = padded_img / 255.0 # normalize the image
        guidanceForFilter = padded_guidance / 255.0 # normalize the guidance image
        width = imageForFilter.shape[1] # get the width of the image
        height = imageForFilter.shape[0] # get the height of the image

        #check guidence image shape (check if its a gray image or not)
        guidenceIsGray = False
        if len(np.squeeze(guidance).shape) < 3:
            guidenceIsGray = True

        #create spatial kernel
        #the spatial kernel is a 3D array, with the shape of (window_H, window_W, 3), so we can directly multiply the spatial kernel  with the window
        spatialKernel = np.zeros((self.wndw_size,self.wndw_size, 3))
        for i in range(-self.pad_w, self.pad_w+1):
            for j in range(-self.pad_w, self.pad_w+1):
                spatialKernel[i+self.pad_w ,j+self.pad_w,:] = np.repeat(np.exp(-0.5 * ((i**2)+ (j**2)) / (self.sigma_s ** 2)),repeats=3)

        #range kernel cant be precomputed as it depends on the value of the guidence image

        # strat the convolution
        output = np.zeros_like(img, dtype=np.float64)
        for h in range(self.pad_w , height - self.pad_w):
            for w in range(self.pad_w, width - self.pad_w):
                #get the window for the current pixel(h,w)
                #the window is a 3D array, with the shape of (window_H, window_W, 3), so we can directly multiply the spatial kernel and range kernel with the window
                window = imageForFilter[h-self.pad_w:h+self.pad_w+1, w-self.pad_w:w+self.pad_w+1]
                #get the guidance window for the current pixel(h,w)
                guidanceWindow = guidanceForFilter[h-self.pad_w:h+self.pad_w+1, w-self.pad_w:w+self.pad_w+1]

                if guidenceIsGray:
                    # if the guidance image is gray, then we can directly calculate the range kernel
                    rangeKernel = np.repeat(np.exp(-0.5 * ((guidanceWindow - guidanceForFilter[h,w])**2)/(self.sigma_r**2))[:,:,np.newaxis],axis=2,repeats=3)
                else:
                    # if the guidance image is not gray, then we need to sum up the difference of each channel,and then calculate the range kernel
                    rangeKernel =  np.repeat(np.exp(-0.5 * (np.sum((guidanceWindow - guidanceForFilter[h,w])**2, axis= 2))/(self.sigma_r ** 2))[:,:,np.newaxis],axis=2,repeats=3)

                # the range kernel is a 3D array, with the same shape of( window_H, window_W, 3), so we can directly multiply the range kernel with the window
                # normalize factor of the kernels, axis=(0,1) make the normalizeFactor to be an array of shape (3,)
                normalizeFactor = 1 / np.sum(spatialKernel * rangeKernel, axis=(0,1))

                # calculate the output
                output[h-self.pad_w,w-self.pad_w] = np.sum(spatialKernel * rangeKernel * window, axis=(0,1)) * normalizeFactor

        output = output * 255 # denormalize the output
        return np.clip(output, 0, 255).astype(np.uint8) # clip the output to the range of [0,255]

    def joint_bilateral_filterF(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w,
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w,
                                             self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # setup a look-up table for spatial kernel
        LUT_s = np.exp(-0.5 * (np.arange(self.pad_w + 1) ** 2) / self.sigma_s ** 2)
        # setup a look-up table for range kernel
        LUT_r = np.exp(-0.5 * (np.arange(256) / 255) ** 2 / self.sigma_r ** 2)
        # compute the weight of range kernel by rolling the whole image
        wgt_sum, result = np.zeros(padded_img.shape), np.zeros(padded_img.shape)
        for x in range(-self.pad_w, self.pad_w + 1):
            for y in range(-self.pad_w, self.pad_w + 1):
                # method 1 (easier but slower)
                dT = LUT_r[np.abs(np.roll(padded_guidance, [y, x], axis=[0, 1]) - padded_guidance)]
                r_w = dT if dT.ndim == 2 else np.prod(dT, axis=2)  # range kernel weight
                s_w = LUT_s[np.abs(x)] * LUT_s[np.abs(y)]  # spatial kernel
                t_w = s_w * r_w
                padded_img_roll = np.roll(padded_img, [y, x], axis=[0, 1])
                for channel in range(padded_img.ndim):
                    result[:, :, channel] += padded_img_roll[:, :, channel] * t_w
                    wgt_sum[:, :, channel] += t_w
        output = (result / wgt_sum)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, :]
        return np.clip(output, 0, 255).astype(np.uint8)
