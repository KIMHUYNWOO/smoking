from mmdet.apis import inference_detector, init_detector
import mmcv
import numpy as np
import torch
import cv2
import os

def select_person_result(bboxes, labels, segm_result, segms, score):
    new_label = []
    new_bboxes = []
    new_segm_result = [] 
    # select only "person" labels
    for idx, label in enumerate(labels):
    # '0 is for "person" class'
        if label == 0 and bboxes[idx][-1] >= score :
            new_label.append(0)
            new_bboxes.append(bboxes[idx])
            if segm_result is not None and len(labels) > 0:
                new_segm_result.append(segms[idx])
    labels = np.array(new_label)
    bboxes = np.array(new_bboxes)
    if new_segm_result:
        segms = np.array(new_segm_result)
    return bboxes, segms

class Seg_Estimator:
    def __init__(self) -> None:
        self.config = 'detection/configs/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco.py'
        self.seg_checkpoint = 'detection/checkpoints/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_20210628_164719-5bdc3824.pth'
        self.device = 'cuda:0'
        self.model = init_detector(self.config, self.seg_checkpoint, self.device)
        
    def run(self, img, score_thr_2d):
        result = inference_detector(self.model, img) # top-k result
        # show_result_pyplot(self.model, img, result, score_thr=0.3, out_file='./result2.jpg')
        
        bbox_result, segm_result = result
        # labels
        labels = [np.full(bbox.shape[0], i, dtype=np.int32)for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels) # (k,)
        # bboxes
        bboxes = np.vstack(bbox_result) # (k,5) (left, top, right, bottom, confidence)
        
        # semg_result processing 'segm: (N_class, H, W)'
        segms = segm_result
        if segm_result is not None and len(labels) > 0:# non empty
            segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
        
        # select only person class (+ filtering)
        bboxes, segms = select_person_result(bboxes, labels, segm_result, segms, score_thr_2d)
       
        return bboxes, segms
    
def crop_and_save_image(image, bboxes, output_folder, file_name):
    # 이미지를 bounding boxes로 crop하고 저장하는 함수
    # image: 원본 이미지
    # bboxes: bounding boxes 좌표 정보 배열
    # output_folder: crop된 이미지를 저장할 폴더 경로
    
    for i,box in enumerate(bboxes):
        x, y, w, h = box[:4]
        # bounding box를 기준으로 이미지 crop
        cropped_img = image[int(y):int(h), int(x):int(w)]
        output_file_name = f'{file_name}_{i}.png'
        # crop된 이미지를 파일로 저장
        output_path = os.path.join(output_folder, output_file_name)
        cv2.imwrite(output_path, cropped_img)
        
        print(f'Cropped image saved at: {output_path}')
        
if __name__ == '__main__':  
    
    input_folder_name = '0117_20'
    
    #Dataset 폴더 경로
    base_folder_path = 'D:/smoking_dataset/dataset'
    input_folder_path = os.path.join(base_folder_path, input_folder_name)
    
    #Dataset 폴더 안에 있는 이미지 리스트 가져오기
    file_list = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]
    print(file_list)
    
    #output 폴더 생성
    output_folder = 'D:/smoking_dataset/cropped'
    output_folder_path = os.path.join(output_folder, input_folder_name)
    os.makedirs(output_folder_path, exist_ok=False)
    
    for file_name in file_list :    
        # 이미지 로드
        file_path = os.path.join(input_folder_path, file_name)
        img = mmcv.imread(file_path)
        
        #img= np.zeros((10,10,3))
        seg_estimator= Seg_Estimator()
        bboxes, segms = seg_estimator.run(img, score_thr_2d=0.3)
        output_folder = output_folder_path
        output_file_name = f'{input_folder_name}_{file_name}'
        crop_and_save_image(img, bboxes, output_folder, output_file_name)
    
