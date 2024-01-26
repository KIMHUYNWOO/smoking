from mmdet.apis import inference_detector, init_detector
import mmcv
import numpy as np
import torch
import cv2

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

def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    # image는 Numpy 배열 (height, width, channels)
    # boxes는 각각의 bounding box의 좌표 정보를 담은 배열
    # color는 사각형의 색깔을 나타내는 RGB 튜플
    # thickness는 사각형의 선 두께
    
    for box in boxes:
        x, y, w, h = box[:4]
        cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), color, thickness)
    
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    # 테스트할 이미지 파일 경로
    img_path = 'detection/smoking_image/0117_12_1/0257.png'
    # 이미지 로드
    img = mmcv.imread(img_path)
    #img= np.zeros((10,10,3))
    seg_estimator= Seg_Estimator()
    bboxes, segms = seg_estimator.run(img, score_thr_2d=0.7)
    output_image_path = 'detection/result_image.png'
    draw_boxes_on_image(img.copy(), bboxes)
    #cv2.imwrite(output_image_path, img)
    print(bboxes)
  