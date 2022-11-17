import imp
import os
import numpy as np
import cv2

root_path = '/home/yj/desktop/codes/mmocr-0.6.1/demo/'

def visual(features):
    i = 0
    for feature in features:
        # print(feature.shape)
        feature = feature[0, 0, :, :].cpu().numpy()
        # print(feature.shape)
        feature = np.asarray(feature * 255, dtype=np.uint8)
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        dst_path = root_path
        dst_file = os.path.join(dst_path, str(i) + '.png')
        cv2.imwrite(dst_file, feature)
        i += 1

def visualize_feature_map_sum(features):
    k = 0
    for feature in features:
        # feature = feature.squeeze(0)
        c = feature.shape[1]
        feature_combination = []

        for i in range(0, c):
            feature_split = feature.data.cpu().numpy()[0, i, :, :]
            feature_combination.append(feature_split)

        feature_sum = sum(one for one in feature_combination)
        feature_sum = np.asarray(feature_sum * 255, dtype=np.uint8)
        feature_sum = cv2.applyColorMap(feature_sum, cv2.COLORMAP_JET)
        cv2.imwrite(root_path + 'feature_sum_'+str(k)+'.png', feature_sum)
        k += 1