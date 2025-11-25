from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def ensure_gray2d(image):
    """確保影像是灰階二維 (H, W)，去掉多餘的 channel 維度"""
    if sys.version_info >= (3, 12):
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)  # (H, W, 1) -> (H, W)
    return image


def adjust_hsv_lightness_by_percentile(image, low=3, high=97):
    """
    使用百分位數拉伸調整 HSV 的 V 通道，避免極端亮/暗點影響。
    
    Args:
        image: 輸入的 BGR 圖像
        low: 下界百分位數 (預設 3)
        high: 上界百分位數 (預設 97)
    
    Returns:
        adjusted_image: 調整後的 BGR 圖像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 以百分位數決定映射範圍
    vmin = float(np.percentile(v, low))
    vmax = float(np.percentile(v, high))

    if vmax <= vmin:  # 防呆
        v_new = v.copy()
    else:
        v_new = ((v.astype(np.float32) - vmin) / (vmax - vmin) * 255.0)
        v_new = np.clip(v_new, 0, 255).astype(np.uint8)

    adjusted_hsv = cv2.merge((h, s, v_new))
    adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image

def get_plate(model, img):
    result = model.predict(source=img, conf=0.8, save=False, verbose=False)
    found = []
    for res in result:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x2 - x1 > 60:
                cropped_plate = img[y1:y2, x1:x2]
                resized_plate = cv2.resize(cropped_plate, (416, 416), interpolation=cv2.INTER_CUBIC)
                gray_plate = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)

                # 在原圖上畫紅色框 (BGR: (0, 0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

                found.append((gray_plate, (x1, y1, x2, y2)))
    return found    

def get_characters(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = model.predict(source=img, save = False)
    plate_id = ''
    for res in result:
        chars, char_nums, dash = [], 0, 0
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  
            conf = box.conf[0].item()     
            kms = int(box.cls[0].item())  
            class_name = char_model.names[kms]
            #print(class_name, f'{conf:.2f}', end='  ')

            ch = True
            for i, char in enumerate(chars):
                if abs((char[0]+char[2])-(x1+x2)) <= 5:
                    if conf > char[4]:
                        chars[i] = (x1, y1, x2, y2, conf, class_name)
                        ch = False
                        break
                    else:
                        ch = False
                        break
            if ch:
                chars.append([x1.item(), y1.item(), x2.item(), y2.item(), conf, class_name])

        chars.sort(key=lambda x: x[0])

        for i, char in enumerate(chars):
            if char[5] == '-':
                dash = i
                break
        
        while dash>4:    #remove the 5th(and later) to the left of dash
            chars.pop(0) 
            dash = dash - 1
        while len(chars) - dash > 5:    #remove the 5th(and later) to the right of dash
            chars.pop(-1)
          
        if len(chars) - dash == 5 and dash == 4:    #4+4
            chars.pop(0)
            dash = dash - 1
        if len(chars) - dash == 4 and dash == 4:    #4+3
            chars.pop(-1)
            
        if len(chars) - dash == 5 and dash == 3:    #potential 3+4 error
            if chars[0][5] == '1':
                chars.pop(0)
                dash = dash -1
                
        for i, char in enumerate(chars):
            plate_id += char[5]

    return plate_id

# 下半邊緣增強
def edgeup(image):
    # 先確保輸入是正確的 shape
    image = ensure_gray2d(image)

    height, width = image.shape[:2]

    # 切割影像
    top_half = ensure_gray2d(image[:height//2, :])   # 上半部
    bottom_half = ensure_gray2d(image[height//2:, :])  # 下半部

    # 下半強化邊緣
    bottom_half_blurred = ensure_gray2d(cv2.GaussianBlur(bottom_half, (21, 21), 0))
    bottom_half_sharpened = ensure_gray2d(
        cv2.addWeighted(bottom_half, 1.5, bottom_half_blurred, -0.5, 0)
    )

    # 合併部分
    result = np.vstack((top_half, bottom_half_sharpened))
    result = ensure_gray2d(result)

    return result

# 調整queryImage大小，使其與sampleImage大小相似
def zoom_queryImage(sampleImage, queryImage):
    sy, sx = sampleImage.shape[:2]
    qy, qx = queryImage.shape[:2]
    zoomscale = ((sy / qy)+(sx / qx))/2
    # 根據zoomscale縮放queryImage
    new_size = (int(qx * zoomscale), int(qy * zoomscale))  # 計算新的寬度和高度
    queryImage = cv2.resize(queryImage, new_size, interpolation=cv2.INTER_LINEAR)  # 縮放
    return queryImage

# 左右兩上三角塗黑
def black_out_triangles(image):
    height, width = image.shape[:2]
    hscale = height * 2 / 5
    wscale = width / 5
    # 左上角的三角形頂點
    left_triangle = np.array([[0, 0], [wscale, 0], [0, hscale]], np.int32)
    left_triangle = left_triangle.reshape((-1, 1, 2))
    
    # 右上角的三角形頂點
    right_triangle = np.array([[width, 0], [width - wscale, 0], [width, hscale]], np.int32)
    right_triangle = right_triangle.reshape((-1, 1, 2))
    
    # 在圖片上繪製這些三角形並圖成黑色
    cv2.fillPoly(image, [left_triangle], 0)
    cv2.fillPoly(image, [right_triangle], 0)
    
    return image

# 邊緣偵測
def edgedetect(image):
    # 使用高斯模糊去噪
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # 使用Canny邊緣偵測
    edges = cv2.Canny(blurred_image, 100, 200)
    # 十字結構，使其可以往四周膨脹
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    # 進行膨脹操作
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # 將圖像二值化，使得邊緣是255，其他部分是0
    _, dilated_edges = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY)
    dilated_edges = dilated_edges // 255 # 除以255讓他變只有0和1
    
    return dilated_edges

# 匹配度計算
def getMatchNum(matches, ratio, sampleImage, queryImage, sedImage, qedImage):
    count = 0 # 最後相似度的分母
    sheight, swidth = sampleImage.shape[:2]
    qheight, qwidth = queryImage.shape[:2]
    # sampleImage中心點座標
    scenterx = swidth / 2
    scentery = sheight / 2
    qcentery = qheight / 2
    # 兩圖匹配點之xy座標差距閥值
    scalex = ((swidth / 10) + (qwidth / 10)) / 2
    scaley = ((sheight / 10) + (qheight / 10)) / 2
    
    matchesMask = [[0, 0] for i in range(len(matches))] # 繪製比對圖觀察用，最後或許用不到
    matchNum = 0 # 最後相似度的分子
    
    # 觀察用
    a1 = 0 # 紀錄SampleImage匹配點組數
    a2 = 0 # 紀錄通過位置比對的匹配點組數
    a3 = 0 # 紀錄通過閥值比對的匹配點組數
    a4 = 0 # 紀錄通過位置比對和閥值比對的匹配點組數
    
    # 開始計算
    for i, (m, n) in enumerate(matches):
        a1 = a1 + 1
        pt1 = kp1[m.queryIdx].pt  # 第一張圖片中的特徵點位置 (x1, y1)
        pt2 = kp2[m.trainIdx].pt  # 第二張圖片中的特徵點位置 (x2, y2)
        dis = 1 # 用來乘上比重
        seat = False # 是否通過位置比對
        check = False # 是否通過閥值比對
        
        # 匹配點是否在圖片上半部
        if (pt1[1] < scentery or pt2[1] < qcentery): # 是，以邊緣偵測取到的0/1圖(+1)做為基礎比重
            dis = dis * (sedImage[int(pt1[1])][int(pt1[0])] * qedImage[int(pt2[1])][int(pt2[0])] + 1)
        else: # 否，以1.5做為基礎比重
            dis = dis * 1.5
        
        if ((abs(pt1[0] - pt2[0]) < scalex) and (abs(pt1[1] - pt2[1]) < scaley)): # 位置比對
            dis = dis * 2 # 若通過，比重再乘上二
            a2 = a2 + 1
            seat = True
        if (m.distance < ratio * n.distance):  # 閥值比對
            a3 = a3 + 1
            dis = dis * 2 # 若通過，比重再乘上二
            check = True
        if (seat == True or check == True):
            if (seat == True and check == True):
                dis = dis * 5 # 若兩個皆通過，比重再乘上五
                matchesMask[i] = [1, 0]
                a4 = a4 + 1
            matchNum = matchNum + dis
            count = count + dis
        else:
            count = count + dis
            
    print("SampleImage匹配點組數:", a1)
    print("通過位置比對的匹配點組數:", a2)
    print("通過閥值比對的匹配點組數:", a3)
    print("通過位置比對和閥值比對的匹配點組數:", a4)
    
    return (matchNum, matchesMask, count)

def calculate_histogram_similarity(img1, img2):
    # 取得兩張圖片的尺寸與中心點
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    center_x1, center_y1 = w1 // 2, h1 // 2
    center_x2, center_y2 = w2 // 2, h2 // 2

    # 切出中間的區域
    img1 = img1[:, center_x1 - (w1 // 4):center_x1 + (w1 // 4)]
    img2 = img2[:, center_x2 - (w2 // 4):center_x2 + (w2 // 4)]
    
    # 調整亮度（使用百分位數拉伸）
    img1 = adjust_hsv_lightness_by_percentile(img1, low=3, high=97)
    img2 = adjust_hsv_lightness_by_percentile(img2, low=3, high=97)
    
    '''
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    '''
    '''
    # 計算每個通道的直方圖
    colors = ['b', 'g', 'r'] 
    plt.figure(figsize=(10, 7))
     # 顯示 img1 和 img2 的直方圖
    for i, color in enumerate(colors):
        # 計算直方圖
        hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
        
        # 把直方圖正歸化
        hist1 = cv2.normalize(hist1, hist1, norm_type=cv2.NORM_L1).flatten()
        hist2 = cv2.normalize(hist2, hist2, norm_type=cv2.NORM_L1).flatten()
    
        # 在同一張圖上畫出兩張圖片的直方圖
        plt.subplot(2, 3, i + 1)
        plt.plot(hist1, color=color, label='Image 1')
        plt.plot(hist2, color=color, linestyle='--', label='Image 2')
        plt.title(f'{color.upper()} Channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()
    '''
    # 計算色彩相似度
    score = 0
    for i in range(3):  # B=0, G=1, R=2
        hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])

        # 正歸化
        hist1 = cv2.normalize(hist1, hist1, norm_type=cv2.NORM_L1).flatten()
        hist2 = cv2.normalize(hist2, hist2, norm_type=cv2.NORM_L1).flatten()

        # Bhattacharyya 距離
        dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        score += (1 - dist) * 100 

    # 平均三通道相似度
    return score / 3

if __name__ == '__main__':
    plate_model = YOLO('plate.pt')
    char_model = YOLO('characters.pt')
    model = YOLO("best.pt")  # 實際的模型權重路徑

    image_path = 'fake/001.png'  # 單張圖片路徑
    if not os.path.exists(image_path):
        print("圖片不存在")
    else:
        img = cv2.imread(image_path)
        
        # 調整亮度（使用百分位數拉伸）
        lightimg = adjust_hsv_lightness_by_percentile(img, low=3, high=97)
        
        # 使用 YOLO 進行物件偵測
        results = model(lightimg, conf=0.8, save=False, verbose=False)

        # 取得偵測框並裁切
        for i, box in enumerate(results[0].boxes.xyxy):  # 取得每個 bounding box
            x1, y1, x2, y2 = map(int, box)  # 轉為整數座標
            cropped = img[y1:y2, x1:x2]  # 裁切影像
            found = get_plate(plate_model, cropped)  # 找出車牌
        
        if not found:
            print("辨識不出")
        else:
            plate_id = get_characters(char_model, found[0][0])  # 只處理第一個車牌
            if plate_id:
                print(f"辨識結果: {plate_id}")
                same_image_path = f"sample/{plate_id}.jpg"
                if os.path.exists(same_image_path):
                    sampleImage = cv2.imread(same_image_path,0)
                    queryImage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    # 調整queryImage大小
                    queryImage = zoom_queryImage(sampleImage, queryImage)

                    # 取得sampleimage之邊緣偵測圖片
                    sedImage = edgedetect(sampleImage)
                    qedImage = edgedetect(queryImage)

                    # 塗黑三角形區域
                    sampleImage = black_out_triangles(sampleImage)
                    queryImage = black_out_triangles(queryImage)

                    # 邊緣偵測
                    sampleImage = edgeup(sampleImage)
                    queryImage = edgeup(queryImage)

                    # 建立SIFT特徵提取器
                    sift = cv2.SIFT_create()
                    # 建立FLANN匹配對象
                    FLANN_INDEX_KDTREE = 0
                    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    searchParams = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

                    # 提取特徵
                    kp1, des1 = sift.detectAndCompute(sampleImage, None)  # 提取圖庫圖片特徵
                    kp2, des2 = sift.detectAndCompute(queryImage, None)  # 提取比對的圖片特徵

                    # 使用FLANN進行特徵點匹配
                    matches = flann.knnMatch(des1, des2, k=2)

                    # 計算匹配的數量
                    matchNum, matchesMask, count = getMatchNum(matches, 0.85, sampleImage, queryImage, sedImage, qedImage)
                    matchRatio = matchNum * 100 / count
                    print("分子:", matchNum)
                    print("分母", count)
                    print(f"相似度:{matchRatio}%")
                    
                    #顏色比對
                    img1 = cv2.imread(same_image_path)
                    img2 = zoom_queryImage(img1, cropped)
                    colorRatio = calculate_histogram_similarity(img1, img2)
                    print(f"色彩相似度:{colorRatio}%")
                    
                    # 繪製匹配結果圖(觀察用，最後或許用不到)
                    drawParams = dict(matchColor=(0, 255, 0),
                                      singlePointColor=(255, 0, 0),
                                      matchesMask=matchesMask,
                                      flags=0)
                    comparisonImage = cv2.drawMatchesKnn(sampleImage, kp1, queryImage, kp2, matches, None, **drawParams)

                    # 顯示匹配結果
                    plt.figure(figsize=(10, 6))
                    plt.imshow(comparisonImage)
                    plt.title('Feature Matching Result (Similarity %.2f%%)' % matchRatio)
                    plt.show()
                    
                else:
                    print(f"找不到對應的圖片: {same_image_path}")
            else:
                print("辨識不出")
#yo