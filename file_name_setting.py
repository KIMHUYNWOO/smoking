import cv2
import os


if __name__ == '__main__':
    base_folder_path = 'D:/smoking_dataset/cropped/smoking'
    file_list = [f for f in os.listdir(base_folder_path) if os.path.isfile(os.path.join(base_folder_path, f))]
    
    i = 0
    for file_name in file_list:
        # Generate the new file name using the variable i
        new_file_name = f"{i}_0.png"
        
        # Construct the full paths for old and new file names
        old_file_path = os.path.join(base_folder_path, file_name)
        new_file_path = os.path.join(base_folder_path, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        
        # Increment the counter
        i += 1