from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


def get_data(train_size = 0.8) : 
    class CustomDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.annotations = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]))
            image = Image.open(f'{img_name}.png')

            label = int(self.annotations.iloc[idx, 1])

            if self.transform:
                image = self.transform(image)

            return image, label

    # 예시: 데이터셋 및 변환 정의
    train_csv_file = '/home/cv-05/hy/detection/dataset/label/train_label.csv'
    test_csv_file = '/home/cv-05/hy/detection/dataset/label/test_label.csv'
    root_dir = '/home/cv-05/hy/detection/dataset/image'
    
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 데이터셋 인스턴스 생성
    
    custom_train_dataset = CustomDataset(csv_file=train_csv_file, root_dir=root_dir, transform=transform_test) #custom_dataset(imgae, label) shape=([3, 224, 224], 1) *488
    custom_test_dataset = CustomDataset(csv_file=test_csv_file, root_dir=root_dir, transform=transform_test)
    
    # 데이터로더 설정
    batch_size = 32
    train_loader = DataLoader(dataset=custom_train_dataset, batch_size=batch_size, shuffle=True)
    teat_loader = DataLoader(dataset=custom_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, teat_loader