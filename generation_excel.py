import os
import numpy as np
import pandas as pd
from openpyxl.workbook import Workbook
from shutil import copyfile

if __name__ == '__main__':
    based_path = 'D:/smoking_dataset/cropped/dataset_real'
    all_dataset_path = os.path.join(based_path, 'all')
    all_dataset_list = [f for f in os.listdir(all_dataset_path) if os.path.isfile(os.path.join(all_dataset_path, f))]

    label_arr = []
    i = 0
    folder_path = os.path.join(based_path, 'label')
    os.makedirs(folder_path, exist_ok=True)
    
    for file_name in all_dataset_list:
        # Generate the new file name using the variable i
        label = file_name
        split_label = label.split('_')
        data_num = i
        right_value = split_label[1].split('.')[0] if len(split_label) > 1 else None

        # Create a folder based on the right_value
        # Construct the full paths for old and new file names
        new_file_name = f"{i}.png"
        old_file_path = os.path.join(all_dataset_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Copy the file to the new folder
        copyfile(old_file_path, new_file_path)

        # Increment the counter
        i += 1

        # Append information to label_arr
        label_arr.append([data_num, right_value])

    df = pd.DataFrame(label_arr, columns=['Number', 'Label'])
    csv_file_path = 'D:/smoking_dataset/cropped/dataset_real/label/label.csv'
    df.to_csv(csv_file_path, index=False)
