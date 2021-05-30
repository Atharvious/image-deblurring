# image-deblurring
A Generative Adverserial approach to sharpen images, using the Pix2Pix GAN model.

## This is a direct implementation of the Pix2Pix model, the respective paper can be found [here](https://arxiv.org/abs/1611.07004).

### Dependencies
Download everything by cloning into the repo or downloading it, run pip install -r requirements.txt. This installs:
* numpy
* pytoch
* tqdm
* albumentations
* pillow




## Using the trained model

* #### Trained weights can be found [here](https://dl.dropbox.com/s/1lykfe2xkcfloz5/Trained-Model.rar?dl=0).
* Download, extract, and paste the generator.tar file in the root folder.
* test.py can be utilized to deblur images in the following ways:
  1. No arguments: If test.py is run without passing any argument, a 'test.jpg' file will be looked for in the root folder which will then be deblurred. 
  2. Specific path of image: If you pass the full path to the image (including the name), the deblurred image will be generated in the same folder.
  3. Batch Testing: If 'batch' is passed as argument, all the images present in 'dataset/test/ld' will be processed and deblurred. This is useful for evaluating the model and getting metrics.
