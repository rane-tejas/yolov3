# from __future__ import annotations
from pickletools import uint8
import numpy as np
# from bs4 import BeautifulSoup
import cv2
import ipdb
import os
import distutils

from natsort import natsorted

# crop_image_x:
# - 0 #293
# - 258 #803
# crop_image_y:
# - 258 #134
# - 516 #637
# Crop parameters for FUKudA probe
CROP_IMAGE_X = [293, 803]
CROP_IMAGE_Y = [134, 637]

def crop_image(img):
    return img[CROP_IMAGE_Y[0]:CROP_IMAGE_Y[1],CROP_IMAGE_X[0]:CROP_IMAGE_X[1]]

def check_boxes_within_image_limts(box, W, H):
    '''
    update box if limit exceeds width, height of cropped image
    '''
    # box0 = box[0] # smaller point
    # box1 = box[1] # bigger point

    box[0][0] = 0 if box[0][0] < 0 else box[0][0]
    box[0][1] = 0 if box[0][1] < 0 else box[0][1]
    box[1][0] = W if box[1][0] > W else box[1][0]
    box[1][1] = H if box[1][1] > H else box[1][1]    

    return box 
    
def extract_points_as_numbers(points_str):
    '''
    convert points from str to array to find min and max of x and y
    '''
    tmp = points_str.split(';')
    tmp_ = [tmp[i].split(',') for i in range(len(tmp))]
    # tmp_ = [[float(tmp_[i][0]), float(tmp_[i][1])] for i in range(len(tmp))]
    # convert to INT and subtract by crop 
    tmp_ = [[int(float(tmp_[i][0]))-CROP_IMAGE_X[0], int(float(tmp_[i][1]))-CROP_IMAGE_Y[0]] for i in range(len(tmp))] #making int so that easier to draw bounding box

    # find min and max along x and y
    tmp_arr = np.array(tmp_)

    min_ = np.min(tmp_arr, axis=0)
    max_ = np.max(tmp_arr, axis=0)

    # ipdb.set_trace()

    return [min_, max_]
    # return tmp_arr

def read_xml(file):

    with open(file,'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "xml")

    # get all polygons
    b_polygon = bs_data.find_all('polygon')

    # store frame number and points for each polygon
    frame_numbers = [int(b_polygon[i].get('frame')) for i in range(len(b_polygon))]
    polygon_points = [b_polygon[i].get('points') for i in range(len(b_polygon))]
    bounding_boxes = [extract_points_as_numbers(polygon_points[i]) for i in range(len(b_polygon))]
    # ipdb.set_trace()

    return bounding_boxes, frame_numbers

def img_save(folder_name):
    '''
    save image as frame_number.txt
    '''
    pass

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)   
        
def read_video_save_image_with_label(video, frame_numbers, bounding_boxes, txt_folder='dataset/vessels/labels',img_folder='dataset/vessels/images'):
    '''
    read mp4 file and save images which have frame numbers
    '''

    # create folders if not present
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    cap = cv2.VideoCapture(video)
    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_list = []
    cnt = 0 # keep frame count 
    while(cap.isOpened() and cnt < totalframecount):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_cropped = crop_image(frame)
        frames_list.append(frame_cropped)
        cv2.imwrite(os.path.join(img_folder, str(cnt) + '.png'), frame_cropped)
        # check if crop correct:        
        if cnt in frame_numbers:
            # cv2.imwrite('img_og.png', frame)            
            # cv2.imwrite('img_cropped.png', frame_cropped)
            ## draw bounding box with crop included
            ## ind_ = frame_numbers.index(cnt)
            # ipdb.set_trace()
            ind_ = np.where(np.array(frame_numbers) == cnt)[0]
            # box_ = bounding_boxes[ind_]            

            # # make mask
            # size_ = frame_cropped.shape
            # frame_mask = np.zeros((size_[0],size_[1]), dtype=np.uint8)
            # frame_mask[box_[:,1],box_[:,0]] = 255
            # cv2.imwrite('img_mask.png',frame_mask)

            # # draw contours 
            # contours, hierarchy = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # frame_contours = cv2.drawContours(frame_cropped, contours, -1, (0,255,0), 2)
            # cv2.imwrite('img_contours.png',frame_contours)

            # find labels for cropped image and write in TXT file
            with open(os.path.join(txt_folder,str(cnt)+'.txt'),'w') as f:
                for ind in ind_:
                    box_ = bounding_boxes[ind]
                    # uncomment to debug bounding box problem
                    # frame_cropped = cv2.rectangle(frame_cropped, box_[0], box_[1], color=(255,0,0),thickness=4)
                    size_ = np.shape(frame_cropped)
                    W = size_[1] #np returns swapped axis compared to cv2
                    H = size_[0] #np returns swapped axis compared to cv2
                    
                    # check box limit 
                    box_ = check_boxes_within_image_limts(box_, W, H)

                    box_size = box_[1] - box_[0]
                    box_centre = box_[0] + box_size/2

                    f.write('{} {:4f} {:4f} {:4f} {:4f} \n'.format(0, box_centre[0]/W, box_centre[1]/H, box_size[0]/W, box_size[1]/H))
                    
            # uncomment below to debug bounding box 
            # cv2.imwrite('img_cropped_BB.png',frame_cropped)

        cnt += 1
        print("cnt = " , cnt)
    

    return frames_list

def read_images_masks_save_boxes(CVAT_folder, list_CVAT_folders, save_folder):

    cropped_img_folder = os.path.join(save_folder, 'images')
    cropped_img_folder_BB = os.path.join(save_folder, 'images_BB')
    txt_label_folder = os.path.join(save_folder, 'labels')

    make_folder(cropped_img_folder)
    make_folder(txt_label_folder)
    make_folder(cropped_img_folder_BB)

    for i,name in enumerate(list_CVAT_folders):

        image_folder = os.path.join(CVAT_folder, name, 'JPEGImages')
        mask_folder = os.path.join(CVAT_folder, name, 'SegmentationClass')
        # txt_label_folder = os.path.join(CVAT_folder, name, 'YoloLabel')
        # cropped_img_folder = os.path.join(CVAT_folder, name,'CroppedImages')

        cnt = 0    
        while True:

            image_path = os.path.join(image_folder, 'frame_' + '000{}'.format(cnt).zfill(6) + '.PNG')
            if not os.path.exists(image_path):
                print("no more frames to read")
                break

            # ipdb.set_trace()
            img = cv2.imread(image_path)
            mask_path = os.path.join(mask_folder, 'frame_' + '000{}'.format(cnt).zfill(6) + '.png') 
            mask = cv2.imread(mask_path, 0)

            ### FIND BOXES VIA COUNTOUR FINDING
            contour_points, _ = cv2.findContours(mask, 1, 2)

            if len(contour_points) == 0:
                print("no label for image {}".format(cnt))
            else:
                boxes = []
                # img_BB = img
                img_crop_BB = crop_image(img)
                cv2.imwrite(os.path.join(cropped_img_folder, '{}'.format(i+2) +'000{}'.format(cnt).zfill(6) + '.png'), img_crop_BB)
                size_ = np.shape(img_crop_BB)
                H, W = size_[1], size_[0]
                # ipdb.set_trace()
                # cv2.imwrite('img_crop.png', img_crop_BB)
                for contours in contour_points:
                    max_pt = np.max(contours[:,0,:], axis=0)
                    min_pt = np.min(contours[:,0,:], axis=0)
                
                    ### ADJUST BOX COORDINATE ACCORDING TO CROP
                    min_pt_crop = min_pt - np.array([CROP_IMAGE_X[0], CROP_IMAGE_Y[0]])
                    max_pt_crop = max_pt - np.array([CROP_IMAGE_X[0], CROP_IMAGE_Y[0]])
                    box_ = check_boxes_within_image_limts([min_pt_crop, max_pt_crop], W, H)
                    # plot (UNCOMMENT TO DEBUG)
                    img_crop_BB = cv2.rectangle(img_crop_BB, (min_pt_crop[0], min_pt_crop[1]), (max_pt_crop[0], max_pt_crop[1]), (255,0,0),4)
                    # write for debugging            
                    # cv2.imwrite(os'task_2020-09-24--21-21-07-ultrasound-trial-3-2021_12_15_05_55_34-segmentation mask 1.1'.path.join('test', 'img_crop_BB_{}.png'.format(cnt)), cv2.vconcat([img_crop_BB, crop_image(img)]))
                    # ipdb.set_trace()
                    # cv2.imwrite(os.path.join(cropped_img_folder, '{}'.format(i+2) +'000{}'.format(cnt).zfill(6) + '.png'), img_crop_BB)
                    boxes.append(box_)
                cv2.imwrite(os.path.join(cropped_img_folder_BB, '{}'.format(i+2) +'000{}'.format(cnt).zfill(6) + '.png'), img_crop_BB)
                # ipdb.set_trace()
                # draw box on cropped image and box limit checked image 
                txt_label_path = os.path.join(txt_label_folder, '{}'.format(i+2) +'000{}'.format(cnt).zfill(6) + '.txt')
                with open(txt_label_path,'w') as f:
                    for box_ in boxes:

                        box_size = box_[1] - box_[0]
                        box_centre = box_[0] + box_size/2

                        f.write('{} {:4f} {:4f} {:4f} {:4f} \n'.format(0, box_centre[0]/W, box_centre[1]/H, box_size[0]/W, box_size[1]/H))        

            cnt += 1

def read_images_masks_save_boxes():
    PIG_TRAIN = '/home/tejasr/projects/tracir_segmentation/data/pig_dataset_fukuda/train/'
    PIG_VALID = '/home/tejasr/projects/tracir_segmentation/data/pig_dataset_fukuda/valid/'
    PIG_TEST = '/home/tejasr/projects/tracir_segmentation/data/pig_dataset_fukuda/test/'
    YOLO_TRAIN = '/home/tejasr/projects/tracir_segmentation/data/yolo_dataset/train/'
    YOLO_VALID = '/home/tejasr/projects/tracir_segmentation/data/yolo_dataset/valid/'
    YOLO_TEST = '/home/tejasr/projects/tracir_segmentation/data/yolo_dataset/test/'
    PIG_SET = [PIG_TRAIN, PIG_VALID, PIG_TEST]
    YOLO_SET = [YOLO_TRAIN, YOLO_VALID, YOLO_TEST]

    for src, dst in zip(PIG_SET, YOLO_SET):
        distutils.dir_util.copy_tree(src+'images/', dst+'images/')
        labels = natsorted(os.listdir(src+'labels/'))
        for i, name in enumerate(labels):
            image_path = src+'images/'+name
            image = cv2.imread(image_path)
            label_path = src+'labels/'+name
            label = cv2.imread(label_path, 0)
            contour_points, _ = cv2.findContours(label, 1, 2)
            boxes = []
            size_ = np.shape(image)
            H, W = size_[1], size_[0]
            for contours in contour_points:
                max_pt = np.max(contours[:,0,:], axis=0)
                min_pt = np.min(contours[:,0,:], axis=0)
                box_ = [min_pt, max_pt]
                image_bb = cv2.rectangle(image, (min_pt[0], min_pt[1]), (max_pt[0], max_pt[1]), (255,0,0), 1)
                boxes.append(box_)
            cv2.imwrite(os.path.join(dst+'images_bb', name.split('.')[0]+'.png'), image_bb)
            label_path = os.path.join(dst+'labels', name.split('.')[0]+'.txt')
            with open(label_path,'w') as f:
                for box_ in boxes:
                    box_size = box_[1] - box_[0]
                    box_centre = box_[0] + box_size/2
                    f.write('{} {:4f} {:4f} {:4f} {:4f} \n'.format(0, box_centre[0]/W, box_centre[1]/H, box_size[0]/W, box_size[1]/H))

if __name__ == "__main__":

    read_images_masks_save_boxes()

    # # replace below with your video_name and annotation file name
    # video_name = '2022-05-19--20-30-34-pig_lab-Trial-5.mp4'
    # annotations_file_name = 'annotations.xml'

    # bounding_boxes, frame_numbers = read_xml('annotations.xml')
    # # ipdb.set_trace()
    # frames = read_video_save_image_with_label(video_name, frame_numbers,bounding_boxes)
    # # read video to frames | crop it | save frame using their index +.png

    # # read xml file and create .txt files for given frames 
    # # we'll need to adjust the crop as well 

    # # create bounding boxes using the .txt files to verify 
    
