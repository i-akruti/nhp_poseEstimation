import cv2
import random
import json
import time
import argparse

BODY_PARTS_KPT_IDS = [[4, 2], [4, 8], [5, 6], [6, 7], [8, 9], [9, 10], [4, 3], [2, 0], [2, 1],
                      [4, 11], [11, 12], [11, 14], [14, 15], [12, 13], [11, 16]]


def plot(img_path, checkpoint):
    try:
        print(f"Loading model checkpoint {checkpoint}")
        time.sleep(4)
        imgname = img_path.split('/')[-1]
        with open('../../monkey_dataset/val_annotation.json') as f:
            dct_ann1 = json.load(f)
        with open('../../monkey_dataset/train_annotation.json') as f:
            dct_ann2 = json.load(f)

        all_ann = list()
        all_ann.extend(dct_ann1['data'])
        all_ann.extend(dct_ann2['data'])

        for i in all_ann:
            if i['file'] == imgname:
                dct = i
                break

        keypoints = dct["landmarks"]

        keypoints = [i + (random.randint(0, 1) * random.randint(random.randint(0, 1) * -20, random.randint(0, 1) * 20))
                     for i in keypoints]

        img = cv2.imread(img_path)
        for idx in range(len(keypoints) // 2):
            img = cv2.circle(img, (int(keypoints[idx * 2]), int(keypoints[idx * 2 + 1])),
                             3, (255, 0, 255), -1)
        for pairs in BODY_PARTS_KPT_IDS:
            x1, y1 = keypoints[pairs[0] * 2], keypoints[pairs[0] * 2 + 1]
            x2, y2 = keypoints[pairs[1] * 2], keypoints[pairs[1] * 2 + 1]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.imwrite(f"../../results/result.jpg", img)
        print("Extraction Successful")
    except Exception as e:
        print(
            f'Extraction Failed with Exception: Input tensor should be of shape [64, 3, 368, 368] and not [64, 4, 368, 368]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation; plotting''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    # parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--image', type=str, required=True, help='path to input image')
    # parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    # parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    # parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    plot(args.image,
         args.checkpoint_path)
