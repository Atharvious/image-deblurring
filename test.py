from generator import Generator
import config
import torch
from utils import load_checkpoint
import torch.optim as optim
from dataset import DeblurDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image
import sys

#Since the DeblurDataset class is catered for batch images,
#It's easier to load a single image directly, and process accordingly
def single_test(img_path):
    
    #initialize generator and load pre-trained weights 
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5,0.999)) 
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    
    #load the image
    input_image = np.array(Image.open(img_path))
    
    #crop margins
    input_image = input_image[config.MARGIN_WIDTH:-config.MARGIN_WIDTH,config.MARGIN_WIDTH:-config.MARGIN_WIDTH,:]
    
    #normalize pixel values, 
    #change dimensions to 'channel first',
    #and convert to tensor
    input_image = config.resize_and_normalize(image=input_image)["image"]
    
    #Expand dims to the input tensor - tensor dims should be [1,3,256,256] after excecution of following line
    input_image = torch.unsqueeze(input_image, 0)
    
    input_image = input_image.to(config.DEVICE)
    gen.eval()
    
    
    with torch.no_grad():
        generated_image = gen(input_image)
        
        #Undo Normalization
        generated_image = generated_image * 0.5 + 0.5
        
        #Save the output image
        save_image(generated_image, img_path[:-4] + "_deblurred.png")




#Batch_Testing
def test(gen, loader, folder,run):
    input_image, target_image = next(iter(loader))
    input_image, target_image = input_image.to(config.DEVICE), target_image.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        generated_image = gen(input_image)
        generated_image = generated_image
        grid = torch.cat([input_image * 0.5 + 0.5, generated_image * 0.5 + 0.5])
        save_image(grid , os.path.join(folder,f"deblurred_{run+1}.png"))

        
def batch_test():
    #initialize generator
    gen = Generator(in_channels=3).to(config.DEVICE)
    
    #optimizer isn't required but the load_checkpoint function loads it,
    #so it's easier to just initialize it as well
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5,0.999))
    
    #Load the generator
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    
    #Initialize the dataflow
    test_dataset = DeblurDataset(root_dir=config.TEST_DIR)
    print("Number of Images: ", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size= len(test_dataset))
    
    #Run the images through the generator model
    for run in range(config.NUM_RUNS):
        test(gen, test_loader, "test_results/", run)
        print("Gen {} complete.".format(run+1))
        
if __name__ == "__main__":
    #main()
    if len(sys.argv) == 1:
        try:
            single_test("test.jpg")
        except FileNotFoundError:
            print("Please make sure there is 'test.jpg' in the root folder")
            print("Or if you want to specify the file path, run the script from commandline and pass the image filepath(including the filename) as the argument")
            
    else:
        if sys.argv[1] == "batch":
            batch_test()
        else:
            single_test(sys.argv[1])
        
    
