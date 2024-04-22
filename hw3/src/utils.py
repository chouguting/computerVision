import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    #一個pair會形成 2x9 的矩陣
    #N個pair會形成 2N x 9 的矩陣
    A = np.zeros((2 * N, 9))
    for i in range(N):
        #u是source pixel location
        #u[i]的homo coordinate是(u[i][0], u[i][1], 1)
        #v是destination pixel location
        #v[i]的homo coordinate是(v[i][0], v[i][1], 1)
        #每一組pair會形成 2x9 的矩陣
        #長的像這樣:
        # [[ 0^T, -1*u^T, v[i][1]*u^T ],
        #  [ u^T, 0^T, -v[i][0]*u^T ]]
        # =
        # [ [0, 0, 0, -1*u[i][0], -1*u[i][1], -1*1, v[i][1]*u[i][0], v[i][1]*u[i][1], v[i][1]],
        #   [1*u[i][0], 1*u[i][1], 1*1, 0, 0, 0, -v[i][0]*u[i][0], -v[i][0]*u[i][1], -v[i][0]] ]
        A[2 * i] = np.array([0, 0, 0, -1 * u[i][0], -1 * u[i][1], -1, v[i][1] * u[i][0], v[i][1] * u[i][1], v[i][1]])
        A[2 * i + 1] = np.array([u[i][0], u[i][1], 1, 0, 0, 0, -v[i][0] * u[i][0], -v[i][0] * u[i][1], -v[i][0]])

    # TODO: 2.solve H with A
    #用SVD解
    #A = U S V^T, 此時V的最後一行就是h, h是H的flatten
    h = np.linalg.svd(A)[2][-1]
    H = h.reshape(3, 3)
    return H




def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    #把所有(X,Y) pair都集合起來
    #meshgrid會把所有的XY組合都列出來，並將X和Y分別放在不同的矩陣
    sourceXs, sourceYs = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    #把所有homogeneous coordinate都集合起來
    srcPointsHomogeneous = np.ones((sourceXs.size, 3))
    srcPointsHomogeneous[:, 0] = sourceXs.flatten()  #flatten會把矩陣拉平
    srcPointsHomogeneous[:, 1] = sourceYs.flatten()  #flatten會把矩陣拉平

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        #在backward warping中，srcPointsHomogeneous是destination image的pixel的homogeneous coordinate
        #H_inv是destination image到source image的轉換矩陣, 形狀是3x3
        #srcPointsHomogeneous是destination image的pixel的homogeneous coordinate, 形狀是Nx3
        #我們計算 H_inv * srcPointsHomogeneous.T, 也就是3x3 * 3xN = 3xN
        backWarpedPoints = np.dot(H_inv, srcPointsHomogeneous.T).T  #經過Transpose後，會得到Nx3的矩陣
        backWarpedPoints = backWarpedPoints * 1.0 / (backWarpedPoints[:,2].reshape(-1,1)) #將homogeneous coordinate轉換成cartesian coordinate
        #把所有的source point的x座標和y座標分開，並reshape成和destinaton box一樣的形狀
        #(sourceX[a], sourceY[b]) 的值代表的是destination image的pixel (a, b) 在source image的位置
        #先不要轉換成int，因為我們等一下要做內插
        sourceXs = backWarpedPoints[:, 0].reshape(ymax - ymin, xmax - xmin) #backward warped的x座標(在source image的位置)
        sourceYs = backWarpedPoints[:, 1].reshape(ymax - ymin, xmax - xmin)


        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        #邊界的mask
        #要取的點不能超過source image的邊界
        #假設validPointsMask(a,b) = True，代表destination image的pixel (a, b) 在source image中能找到對應的pixel
        #因為我們內差時要用floor計算與左上角的距離，所以不希望剛好在邊界上(w_src-1, h_src-1)，不然計算內差時它的右下角會沒有值
        validPointsMask = ((sourceXs >= 0) & (sourceXs < w_src-1)) & ((sourceYs >= 0) & (sourceYs < h_src-1))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        #只取在邊界內的點
        validSourceXs = sourceXs[validPointsMask]
        validSourceYs = sourceYs[validPointsMask]


        # TODO: 6. assign to destination image with proper masking
        #用線性內插的方式，把source image的pixel值assign到destination image
        #先無條件捨去，取左上角的pixel
        validSourceXsFloor = np.floor(validSourceXs).astype(int)
        validSourceYsFloor = np.floor(validSourceYs).astype(int)

        #計算距離左上角的距離
        deltaX = (validSourceXs - validSourceXsFloor).reshape(-1, 1) #把shape變成Nx1(本來是N)
        deltaY = (validSourceYs - validSourceYsFloor).reshape(-1, 1)

        #計算內差
        #左上角的pixel值
        topLefts = src[validSourceYsFloor, validSourceXsFloor, :]
        #右上角的pixel值
        topRights = src[validSourceYsFloor, validSourceXsFloor + 1, :]
        #左下角的pixel值
        bottomLefts = src[validSourceYsFloor + 1, validSourceXsFloor, :]
        #右下角的pixel值
        bottomRights = src[validSourceYsFloor + 1, validSourceXsFloor + 1, :]
        #實際內插的值
        interpolatedValues = topLefts * (1 - deltaX) * (1 - deltaY) + topRights * deltaX * (1 - deltaY) + bottomLefts * (1 - deltaX) * deltaY + bottomRights * deltaX * deltaY
        #把內插的值assign到destination image
        #先把desination的min~max區域框出來
        dst[ymin:ymax,xmin:xmax][validPointsMask] = interpolatedValues


    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        #計算source image的pixel在destination image的位置
        #H是source image到destination image的轉換矩陣, 形狀是3x3
        #srcPointsHomogeneous是source image的pixel的homogeneous coordinate, 形狀是Nx3
        #我們計算 H * srcPointsHomogeneous.T, 也就是3x3 * 3xN = 3xN
        destPoints = np.dot(H, srcPointsHomogeneous.T).T  #經過Transpose後，會得到Nx3的矩陣
        destPoints = destPoints * 1.0 / (destPoints[:,2].reshape(-1,1)) #將homogeneous coordinate轉換成cartesian coordinate
        #把所有的destination point的x座標和y座標分開，並reshape成和source image一樣的形狀
        #(destinationX[a], destinationY[b]) 的值代表的是source image的pixel (a, b) 在destination image的位置
        destinationXs = destPoints[:, 0].reshape(ymax - ymin, xmax - xmin).astype(int)
        destinationYs = destPoints[:, 1].reshape(ymax - ymin, xmax - xmin).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        #邊界的mask
        #(destinationXs >= 0)會產生一個boolean mask, 代表哪些點的x座標在邊界內
        #(destinationXs < w_dst)會產生一個boolean mask, 代表哪些點的x座標在邊界內
        #把他們用&連接起來，就可以得到哪些點的x座標在邊界內 (bit wise and)
        #這個mask的形狀會和source image一樣
        validPointsMask = ((destinationXs >= 0) & (destinationXs < w_dst)) & ((destinationYs >= 0) & (destinationYs < h_dst))

        # TODO: 5.filter the valid coordinates using previous obtained mask
        #只取在邊界內的點
        validDestinationXs = destinationXs[validPointsMask]
        validDestinationYs = destinationYs[validPointsMask]

        # TODO: 6. assign to destination image using advanced array indicing
        #把合法點的source image的pixel值assign到destination image
        #dst(高度, 寬度, 通道) = src(合法點, 通道)
        #高度是y座標，寬度是x座標
        dst[validDestinationYs,validDestinationXs, :] = src[validPointsMask, :]

    return dst

