import numpy as np
import cv2
import copy
import os

plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差

def decodePlate(preds):  # 识别后处理
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    plate = ""
    for i in newPreds:
        plate += plateName[int(i)]
    return plate
    # return newPreds

def rec_pre_precessing(img, size=(48, 168)):  # 识别前处理
    img = cv2.resize(img, (168, 48))
    img = img.astype(np.float32)
    img = (img / 255 - mean_value) / std_value  # 归一化 减均值 除标准差
    img = img.transpose(2, 0, 1)  # h,w,c 转为 c,h,w
    img = img.reshape(1, *img.shape)  # channel,height,width转为batch,channel,height,channel
    return img

def get_plate_result(img, session_rec):  # 识别后处理
    img = rec_pre_precessing(img)
    y_onnx = session_rec.run([session_rec.get_outputs()[0].name], {session_rec.get_inputs()[0].name: img})[0]
    # print(y_onnx[0])
    index = np.argmax(y_onnx[0], axis=1)  # 找出概率最大的那个字符的序号
    # print(y_onnx[0])
    plate_no = decodePlate(index)
    # plate_no = decodePlate(y_onnx[0])
    return plate_no

def allFilePath(rootPath, allFIleList):  # 遍历文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)

def get_split_merge(img):  # 双层车牌进行分割后识别
    h, w, c = img.shape
    img_upper = img[0:int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img

def order_points(pts):  # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):  # 透视变换得到矫正后的图像，方便识别
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def my_letter_box(img, size=(640, 640)):  #
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top

def xywh2xyxy(boxes):  # xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    xywh = copy.deepcopy(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xywh

def my_nms(boxes, iou_thresh):  # nms
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(boxes, r, left, top):  # 返回原图上面的坐标
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top

    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes

def detect_pre_precessing(img, img_size):  # 检测前处理
    img, r, left, top = my_letter_box(img, img_size)
    # cv2.imwrite("1.jpg",img)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
    img = img / 255
    img = img.reshape(1, *img.shape)
    return img, r, left, top

def post_precessing(dets, r, left, top, conf_thresh=0.3, iou_thresh=0.5):  # 检测后处理
    choice = dets[:, :, 4] > conf_thresh
    dets = dets[choice]
    dets[:, 13:15] *= dets[:, 4:5]
    box = dets[:, :4]
    boxes = xywh2xyxy(box)
    score = np.max(dets[:, 13:15], axis=-1, keepdims=True)
    index = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
    output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)
    reserve_ = my_nms(output, iou_thresh)
    output = output[reserve_]
    output = restore_box(output, r, left, top)
    return output

def draw_det_img(img, img_rect):
    img_result = cv2.rectangle(img, (int(img_rect[0]), int(img_rect[1])), (int(img_rect[2]), int(img_rect[3])),
                               (0, 255, 0), 2)
    return img_result

def cut_license(img, img_rect):
    img_result = cv2.rectangle(img, (int(img_rect[0]), int(img_rect[1])), (int(img_rect[2]), int(img_rect[3])),
                               (0, 255, 0), 2)
    img_cut = img_result[int(img_rect[1]):int(img_rect[3]), int(img_rect[0]):int(img_rect[2])]
    return img_cut

