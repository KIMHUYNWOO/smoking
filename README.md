Resnet_8.py : 흡연 판단 모델1(현재 가장 좋은 성능)

Resnet_14.py : 흡연 판단 모델2

Resnet_24.py : 흡연 판단 모델3

data_loader.py : excel파일을 기준으로 이미지를 불러오고 batch_size만큼 데이터를 나누는 dataloader를 return하는 코드

detection_bounding_box.py : human detection을 통해 bounding box를 만들고 시각화 한 코드(필요없음)

detection_crop.py : human detection을 통해 boundig box를 만들고 boundig box를 기준으로 이미지를 잘라주는 코드

main.py : data_loader를 통해 train 이미지를 불러와서 흡연 모델 판단 모델을 통해 train하고 test를 진행하는 코드

file_name_setting.py : 폴더에 있는 이미지의 이름을 0번부터 순서대로 바꿔주는 코드

genration_excel.py : 이미지의 이름을 기준으로 excel을 생성하는 코드

