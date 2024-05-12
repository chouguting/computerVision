import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    #先在image的周圍加上一圈padding (預設是0)
    #這樣在計算邊緣的cost時，就不會超出image的範圍
    leftImagePadded = np.pad(Il, ((1, 1), (1, 1), (0, 0)), 'constant')  #對於長寬都加上一圈padding
    rightImagePadded = np.pad(Ir, ((1, 1), (1, 1), (0, 0)), 'constant')

    #計算census cost
    #對於每一個pixel，都會計算他的binary pattern(也就是中間和周圍的pixel比較，如果比中間大就是0，比中間小就是1)
    #從最左上角的鄰居開始，繞一圈，所以會得到8個binary值，串在一起就是這個(中間) pixel的binary pattern
    #例如: 如果中間的pixel是4，周圍的pixel由左上開始順時針繞一圈是[5, 8, 1, 1, 2, 7, 3, 5]
    #那這個pixel的binary pattern就是[0, 0, 1, 1, 1, 0, 1, 0]
    # 依照這個方法，我們可以得到整張圖的binary pattern:
    # leftImagePattern = np.zeros((h, w, ch, 8), dtype=np.uint8)
    # rightImagePattern = np.zeros((h, w, ch, 8), dtype=np.uint8)
    # for i in range(1, h + 1):
    #     for j in range(1, w + 1):
    #         for c in range(ch):
    #             #取出這個pixel周圍的值
    #             leftNeighbour = leftImagePadded[i - 1:i + 2, j - 1:j + 2, c]
    #             rightNeighbour = rightImagePadded[i - 1:i + 2, j - 1:j + 2, c]
    #             #計算binary pattern
    #             leftPattern = np.where(leftNeighbour > leftNeighbour[1, 1], 0, 1)  #如果比中間大就是0，否則是1
    #             rightPattern = np.where(rightNeighbour > rightNeighbour[1, 1], 0, 1)  #如果比中間小就是1，否則是0
    #             #將binary pattern的周圍一圈以順時針的順序串在一起成為一個一維的array(共有8個值)
    #             #順序為 (0, 0) -> (0, 1) -> (0, 2) -> (1, 2) -> (2, 2) -> (2, 1) -> (2, 0) -> (1, 0)
    #             leftPattern1D = np.concatenate(
    #                 [leftPattern[0, :], leftPattern[1:, 2], leftPattern[2, 1::-1], leftPattern[1:2, 0]])
    #             rightPattern1D = np.concatenate(
    #                 [rightPattern[0, :], rightPattern[1:, 2], rightPattern[2, 1::-1], rightPattern[1:2, 0]])
    #
    #             #將binary pattern存起來，因為1D的binary pattern有8個binary值，所以可以先存成uint8
    #             leftPatternUint8 = np.packbits(leftPattern1D)
    #             rightPatternUint8 = np.packbits(rightPattern1D)
    #             leftImagePattern[i - 1, j - 1, c] = leftPatternUint8
    #             rightImagePattern[i - 1, j - 1, c] = rightPatternUint8

    #理論上上面的方法是對的，但是因為這個方法太慢了，所以我們可以用下面的方法來加速
    #第一個改變是我們利可以利用np.roll來一次進行整個window的移動 (但這樣就不好跳過中間的pixel，所以只能也將中間的pixel也算進去，不過這不影響結果)
    # (把中間的pixel也算進去，所以會有9個binary值)
    #第二個改變是我們將每個bit分別存起來，這樣就可以直接計算hamming distance (原本我是會先轉成uint8)
    #第三個改變是我們不再用順時針的順序來存bit，而是由左上到右下的順序來存，這樣就可以直接用index來存取
    #總而言之，若現在的pixel是(1, 1),value=4，其他pixel值如下:
    # 5 8 1
    # 5 4 1
    # 3 7 2
    # 那這個pixel的binary pattern就是[0, 0, 1, 0, 0, 1, 1, 0, 1]，
    # 因為左上的5比中間的4大，所以是0，中上的8比中間的4大，所以是0，右上的1比中間的4小，所以是1
    # 左邊的5比中間的4大，所以是0，中間的4和中間的4一樣，所以是0，右邊的1比中間的4小，所以是1
    # 左下的3比中間的4小，所以是1，中下的7比中間的4大，所以是0，右下的2比中間的4小，所以是1
    # 要達成這樣的結果，我們需要將鄰居依照以下個順序來shift，並和自己比較:
    # 右下 -> 下 -> 左下 -> 右 -> 中 -> 左 -> 右上 -> 上 -> 左上
    shiftAmount = [(1, 1), (1, 0), (1, -1), (0, 1), (0, 0), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
    leftImageBinaryPattern = np.zeros((h, w, ch, 9))
    rightImageBinaryPattern = np.zeros((h, w, ch, 9))

    for index in range(9):
        # shift = (i, j)代表要將整個window(也就是整張圖)往下移i，往右移j
        leftNeighbour = np.roll(leftImagePadded, shift=shiftAmount[index], axis=(0, 1))
        rightNeighbour = np.roll(rightImagePadded, shift=shiftAmount[index], axis=(0, 1))
        # 看看左右兩張圖的binary pattern，如果比中間的大就是0，否則是1
        #在這邊的中間指的是原圖的基準點，也就是ImagePadded的每個pixel
        # Neighbour的每個pixel都是ImagePadded的pixel shift過的結果，所以比較Neibour和ImagePadded的pixel就可以得到binary patter
        leftImageBinaryPattern[:, :, :, index] = np.where(leftNeighbour > leftImagePadded, 0, 1)[1:-1, 1:-1,
                                                 :]  #要記得把padding去掉
        rightImageBinaryPattern[:, :, :, index] = np.where(rightNeighbour > rightImagePadded, 0, 1)[1:-1, 1:-1,
                                                  :]  #要記得把padding去掉

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    #這邊一樣太慢了:
    # #先創建一個cost volume，大小為(h, w, max_disp+1)，每一個pixel都有max_disp個cost (三個channel的cost會分開存)
    # #因為之後要做Disparity Optimization，所以左右兩張圖的cost都要存起來
    # leftCostVolume = np.zeros((h, w, ch, max_disp + 1), dtype=np.float32)  #從左圖轉到右圖的disparity cost
    # rightCostVolume = np.zeros((h, w, ch, max_disp + 1), dtype=np.float32)  #從左圖轉到右圖的disparity cost
    #
    # #根據左右兩張圖的binary pattern，計算hamming distance
    # for d in range(max_disp + 1):  #對於每一個disparity(從0到max_disp)
    #     #d代表的是兩圖之間的disparity，
    #     #對於右圖，disparity越大，物體相較於左圖就會越往左，而對於左圖，disparity越大，物體相較於右圖就會越往右
    #     #因此，在求rigth cost的時候，我們會固定左圖中的一個位置，並從右圖中的相同位置開始往作搜尋不同disparity的cost
    #     #相對的，在求left cost的時候，我們會固定右圖中的一個位置，並從左圖中的相同位置開始往右搜尋不同disparity的cost
    #     #這樣就可以求出左右兩張圖之間不同disparity的cost
    #     for i in range(1, h + 1):
    #         for j in range(1, w + 1):
    #             for c in range(ch):
    #                 #取出左右兩張圖的binary pattern，作為固定的anchor pattern
    #                 leftAnchorPattern = leftImagePattern[i - 1, j - 1, c]
    #                 rightAnchorPattern = rightImagePattern[i - 1, j - 1, c]
    #                 leftAnchorPatternUint8 = np.array([leftAnchorPattern], dtype=np.uint8)
    #                 rightAnchorPatternUint8 = np.array([rightAnchorPattern], dtype=np.uint8)
    #
    #                 #取出距離anchor pattern d的左右兩張圖的binary pattern
    #                 #對於右圖，要往左移d，對於左圖，要往右移d
    #                 #如果超出邊界，設為0
    #                 leftDisparityPattern = leftImagePattern[i - 1, j - 1 - d, c] if j - 1 - d >= 0 else 0
    #                 rightDisparityPattern = rightImagePattern[i - 1, j - 1 + d, c] if j - 1 + d < w else 0
    #                 leftDisparityPatternUint8 = np.array([leftDisparityPattern], dtype=np.uint8)
    #                 rightDisparityPatternUint8 = np.array([rightDisparityPattern], dtype=np.uint8)
    #
    #                 #計算hamming distance
    #                 #計算left cost時，要利用右圖的固定點及左圖中位移d的點
    #                 #計算right cost時，要利用左圖的固定點及右圖中位移d的點
    #                 leftCost = np.sum(np.unpackbits(rightAnchorPatternUint8 ^ leftDisparityPatternUint8))
    #                 rightCost = np.sum(np.unpackbits(leftAnchorPatternUint8 ^ rightDisparityPatternUint8))
    #
    #                 #存到cost volume中
    #                 leftCostVolume[i - 1, j - 1, c, d] = leftCost
    #                 rightCostVolume[i - 1, j - 1, c, d] = rightCost

    leftCostVolume = np.zeros((h, w, max_disp + 1), dtype=np.float32)  #從左圖轉到右圖的disparity cost
    rightCostVolume = np.zeros((h, w, max_disp + 1), dtype=np.float32)  #從左圖轉到右圖的disparity cost
    sigma_r, sigma_s, WMF_r = 4, 11, 11
    wndw_size = -1  # calculate window size from spatial kernel
    for d in range(max_disp + 1):

        #另一種方式 用roll的方式來計算cost
        # 對於左圖，要往右移d，右移的意思是說相較於右圖中的基準點的座標值，左圖的座標值要往右移d
        # 要達到這個效果，實際上需要將左圖整個左移d，這樣把移動完的左圖和右圖放在一起時，左圖的座標值正好都是右圖的座標值加上d
        # 例如: 如果右圖的基準點是(5, 5)，d=2，那本來在左圖中我們應該要看的點是(5, 7)，也就是右圖的基準點加上d
        #      但是因為左圖整個左了移d=2，所以如果右圖的基準點是(5, 5)，我們只需要看左圖的(5, 5)就可以了
        leftImageBinaryPatternShiftD = np.roll(leftImageBinaryPattern, -d, axis=1)
        #計算hamming distance
        leftCost = np.sum(leftImageBinaryPatternShiftD.astype(np.uint32) ^ rightImageBinaryPattern.astype(np.uint32), axis=3)
        leftCost = np.sum(leftCost, axis=2).astype(np.float32)  # 把所有channel的cost加起來
        # 先把多出來的部分去掉，再補0回去
        # 切掉的原因是因為左圖已經向左移動d，所以左圖的右邊會有d個pixel的cost是沒意義的 (因此會在右邊切掉d個pixel)
        # 補0的原因是剛剛把左圖向左移動d，所以要將左圖移回來，這樣座標才會是對的 (因此會在左邊補d個pixel)
        leftCost = leftCost[:, :w - d]
        leftCost = np.pad(leftCost, ((0, 0), (d, 0)), 'constant')  #因為在右圖中，我們需要向左邊移動d，所以要在左邊加上d個pixel
        #右圖同理， 要將右圖整個向右移動d，這樣把移動完的右圖和左圖放在一起時，相同的左圖座標對應的就會是舊的右圖座標左移d
        rightImageBinaryPatternShiftD = np.roll(rightImageBinaryPattern, d, axis=1)
        rightCost = np.sum(rightImageBinaryPatternShiftD.astype(np.uint32) ^ leftImageBinaryPattern.astype(np.uint32), axis=3)
        rightCost = np.sum(rightCost, axis=2).astype(np.float32)  # 把所有channel的cost加起來
        rightCost = rightCost[:, d:] #因為將右圖整個向右移動d，所以右圖的左邊會有d個pixel的cost是沒意義的 (因此會在左邊切掉d個pixel)
        rightCost = np.pad(rightCost, ((0, 0), (0, d)), 'constant')  #因為將右圖整個向右移動d，又把左邊切掉d個pixel，所以要在右邊加上d個pixel，讓大小和原圖一樣

        # 從左圖到右圖的cost
        rightCostVolume[:,:,d] = xip.jointBilateralFilter(Il, leftCost, -1, sigma_r, sigma_s)
        # 從右圖到左圖的cost
        leftCostVolume[:,:,d] = xip.jointBilateralFilter(Ir, rightCost, -1, sigma_r, sigma_s)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    #決定disparity的方式是，對於每一個pixel，找出cost最小的disparity (最終我們只要右圖的disparity(相對於左圖))
    #這樣就可以得到disparity map
    # for i in range(1, h + 1):
    #     for j in range(1, w + 1):
    #         labels[i - 1, j - 1] = np.argmin(rightCostVolume[i - 1, j - 1])
    #更快的方法是直接用np.argmin，這樣就不用用for loop
    labels = np.argmin(rightCostVolume, axis=2)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    return labels.astype(np.uint8)



