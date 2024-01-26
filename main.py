from data_loader import get_data 
import torch
from Resnet_8 import ResNet
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm 

def preprocessing(train_data, test_data) :
    return None
    

def train(model, train_loader, optimizer):
    model.train()
    criterion = nn.BCELoss()
    train_correct = 0
    train_target_smoking_count = 0
    train_target_nonsmoking_count = 0
    train_pred_smoking_count = 0
    train_pred_nonsmoking_count = 0
    
    for _, (data, target) in enumerate(tqdm(train_loader)):
        # print(f"batch : {i}/{len(train_loader)})")
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        target = target.view(-1, 1).float()
        
        train_target_smoking_count += target.eq(1).sum().item() #50
        train_target_nonsmoking_count += target.eq(0).sum().item() #50
        train_pred = (output > 0.5).float()
        train_pred_smoking_count += (target.eq(1) & train_pred.eq(1)).sum().item()
        train_pred_nonsmoking_count += (target.eq(0)& train_pred.eq(0)).sum().item()
        
        train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # evaluation
    #print(f"Train Loss : {loss}")
    print(f"Train_smoking : {train_pred_smoking_count}/{train_target_smoking_count} acc:{(train_pred_smoking_count/train_target_smoking_count)*100}")
    print(f"Train_nonsmoking : {train_pred_nonsmoking_count}/{train_target_nonsmoking_count} acc:{(train_pred_nonsmoking_count/train_target_nonsmoking_count)*100}")
    print('[{}] Train_Accuracy: {:.2f}%'.format(epoch, train_accuracy))
    
def evaluate(model, test_loader):
    print("************************************")
    model.eval()
    correct = 0
    target_smoking_count = 0
    target_nonsmoking_count = 0
    pred_smoking_count = 0
    pred_nonsmoking_count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            target = target.view(-1, 1).float()
            
            target_smoking_count += target.eq(1).sum().item() #50
            target_nonsmoking_count += target.eq(0).sum().item() #50
            
            pred = (output > 0.5).float()
            
            pred_smoking_count += (target.eq(1) & pred.eq(1)).sum().item()
            pred_nonsmoking_count += (target.eq(0)& pred.eq(0)).sum().item()
            
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        print(f"Test smoking : {pred_smoking_count}/{target_smoking_count} acc:{(pred_smoking_count/target_smoking_count)*100}")
        print(f"Test nonsmoking : {pred_nonsmoking_count}/{target_nonsmoking_count} acc:{(pred_nonsmoking_count/target_nonsmoking_count)*100}")

         
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_accuracy


if __name__ == '__main__':
    #torch.cuda.empty_cache()  
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    
    EPOCHS = 70
    train_data, test_data= get_data(train_size=0.8)
    
    model = ResNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=0.0005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    for epoch in range(1, EPOCHS + 1):
        optimizer.step()
        train(model, train_data, optimizer)
        test_accuracy = evaluate(model, test_data)
        print('[{}] Test Accuracy: {:.2f}%'.format(epoch, test_accuracy))
    
    output = model