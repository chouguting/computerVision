import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}


########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths):
    '''
    Input : 
        img_paths (N) : list of string of image paths
    Output :
        tiny_img_feats (N, d) : ndarray of resized and then vectorized 
                                tiny images
    NOTE :
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''

    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []

    for img_path in tqdm(img_paths):
        img = Image.open(img_path)  # 開啟圖片檔
        img = img.resize((16, 16))  # 調整圖片大小
        img = np.array(img).astype(np.float32) # 轉換成np.array
        img = img.flatten()  # 壓縮成一維
        #img = img / np.sum(img)  # 正規化
        img = (img - np.mean(img)) / np.std(img)  # 正規化(減去平均值後除以標準差)
        tiny_img_feats.append(img)  # 加入list

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats


#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(img_paths, vocab_size=150):
    '''
    Input : 
        img_paths (N) : list of string of image paths (training)
        vocab_size : number of clusters desired
    Output :
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    NOTE :
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better (to a point) 
           but be slower to compute, you can set vocab_size in p1.py
    '''

    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################

    collected_features = []
    for img_path in tqdm(img_paths):
        img = Image.open(img_path)  # 開啟圖片檔
        img = np.asarray(img).astype(np.float32)   # 轉換成np.array
        descriptors = dsift(img, step=[2, 2], fast=True)[1]  # 取得SIFT特徵
        # random_indices = np.random.choice(descriptors.shape[0], size=1500,
        #                                   replace=False)  # 隨機取500個特徵(replace=False代表不重複取樣)
        # descriptors = descriptors[random_indices]  # 取得500個特徵
        collected_features.append(descriptors)  # 加入list

    # collected_features目前是一個list，裡面有N張圖片的m個特徵，每個特徵有128維，因此collected_features的shape是(N, m, 128)
    # 用vstack將list中的array合併，維度變成(N*m, 128)，再轉換成float32
    np_features = np.vstack(collected_features).astype(np.float32)
    vocab = kmeans(np_features, num_centers=vocab_size, initialization="PLUSPLUS")  # 進行k-means聚類
    # 得到的vocab是k個中心點，每個中心點有128維，因此vocab的shape是(k, 128)
    # k=vocab_size
    return vocab

    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################

    # return vocab
    return None


###### Step 1-b-2
def get_bags_of_sifts(img_paths, vocab):
    '''
    Input :
        img_paths (N) : list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Output :
        img_feats (N, d) : ndarray of feature of images, each row represent
                           a feature of an image, which is a normalized histogram
                           of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    for img_path in tqdm(img_paths):
        img = Image.open(img_path)  # 開啟圖片檔
        img = np.array(img).astype(np.float32)  # 轉換成np.array
        descriptors = dsift(img, step=[2, 2], fast=True)[1]  # 取得SIFT特徵
        ## 計算特徵與中心的距離
        # vocab是k個中心點，每個中心點有128維，因此vocab的shape是(k, 128)
        # descriptors是m個特徵，每個特徵有128維，因此descriptors的shape是(m, 128)，
        # (實際上每張圖片的特徵數量不一定相同，但這裡為了方便說明，假設每張圖片的特徵數量都是m)
        # cdist會計算出 m個128維特徵與k個128維中心的距離，因此distances的shape是(m, k)
        # 例如，distances[2, 5]代表第2個特徵與第5個中心的距離
        distances = cdist(descriptors, vocab)
        # 取得每個特徵最近的中心，會得到m個中心的index，因此nearest_centers的shape是(m,)
        nearest_centers = np.argmin(distances, axis=1)
        # 計算每個中心的出現次數，hist是一個長度為k的array，代表每個中心的出現次數
        hist, _ = np.histogram(nearest_centers, bins=len(vocab))
        #img_feats.append(hist / np.sum(hist))  # 正規化(因為每張圖片的特徵數量不一定相同，因此需要正規化)
        hist_norm = [float(i) / sum(hist) for i in hist]
        img_feats.append(hist_norm)

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################

    return img_feats


################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(train_img_feats, train_labels, test_img_feats):
    '''
    Input : 
        train_img_feats (N, d) : ndarray of feature of training images
        train_labels (N) : list of string of ground truth category for each 
                           training image
        test_img_feats (M, d) : ndarray of feature of testing images
    Output :
        test_predicts (M) : list of string of predict category for each 
                            testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
           'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
           'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    CAT2ID = {v: k for k, v in enumerate(CAT)}

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    test_predicts = []
    k = 7
    for test_img_feature in tqdm(test_img_feats):

        # [test_img_feature]是一個 1 x d 的array，train_img_feats是一個 N x d 的array
        # cdist會計算出 1個d維特徵與N個d維特徵的距離，因此distances的shape是(1, N)，再用squeeze將shape轉換成(N,)
        # distances[ 3]代表第3個這張testing image與第3個training image的距離
        distances = cdist(np.asarray([test_img_feature]), train_img_feats, metric='minkowski', p=0.25).squeeze()  # 計算距離
        nearest_indices = np.argsort(distances)[:k]  # 利用argsort找出距離最小的k個index
        nearest_labels = [train_labels[i] for i in nearest_indices]  # 取得k個最近的label
        votingResult = np.zeros(len(CAT))  # 建立一個長度為15的array，用來記錄每個label出現的次數
        for label in nearest_labels:  # 計算每個label出現的次數
            votingResult[CAT2ID[label]] += 1

        predict_index = np.argmax(votingResult)  # 取得出現次數最多的label的index
        predict_label = CAT[predict_index]

        test_predicts.append(predict_label)  # 加入list

    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################

    return test_predicts
