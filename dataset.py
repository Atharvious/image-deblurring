from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset

class DeblurDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        
        #Due to the data structure of the given dataset, we input all image names from the 'ld' folder
        #When loading, we'll look up the corresponding file in 'hd' folder
        self.files_list = os.listdir(os.path.join(self.root_dir, "ld/"))
    
    
    
    #we need to implement __len__() and __getitem__() methods to use torch.utils.data.DataLoader later,
    #for train and test purpose      
    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, index):
        img_file = self.files_list[index]
        
        inp_img_path = os.path.join(os.path.join(self.root_dir,"ld/"), img_file)
        tar_img_path = os.path.join(os.path.join(self.root_dir,"hd/"), img_file)
        #Load the input image
        input_image = np.array(Image.open(inp_img_path))
        
        #Crop out the margins
        input_image = input_image[config.MARGIN_WIDTH:-config.MARGIN_WIDTH,config.MARGIN_WIDTH:-config.MARGIN_WIDTH,:]
        
        try:
            #For test data, we won't have target images and hence need to handle the error accordingly
            target_image = np.array(Image.open(tar_img_path))
            target_image = target_image[config.MARGIN_WIDTH:-config.MARGIN_WIDTH,config.MARGIN_WIDTH:-config.MARGIN_WIDTH,:]
        except FileNotFoundError:
            target_image = input_image
        
        #Normalize, resize and convert both images to tensor (channels first)
        input_image = config.resize_and_normalize(image=input_image)["image"]
        target_image = config.resize_and_normalize(image=target_image)["image"]

        return input_image, target_image
        
    

    


def test():
    root_dir = "./dataset/train"
    data = DeblurDataset(root_dir)
    print(len(data))   # -- should print num of train images --


if __name__ == '__main__':
    test()