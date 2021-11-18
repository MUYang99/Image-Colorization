# Image-Colorization

This is the final project for the [DD2424 - Deep Learning in Data Science](https://www.kth.se/student/kurser/kurs/DD2424?l=en) course of KTH Royal Institute of Technology.

Supervisor: [Josephine Sullivan, KTH Royal Institute of Technology](https://www.csc.kth.se/~sullivan/)

## Task
Explore and compare different deep learning models for image colorization, three representative methods we choosed are U-Net, DCGAN and NoGAN.

## Methodology & Procedure

### Dataset

CIFAR-10 dataset is chosen for this project, which contains 50000 32\*32 images that can be classified to 10 classes.

In our project, two train-validation data subsets from CIFAR-10 are tested. The first one consists of a number of randomly selected samples. The second subset is selected from a sorted set by mean hue value of all the pixels. In this way, generating sets according to a specific proportion so that they all contain similar color distributions and can effectively avoid biasing to any specific color. The chosen validation set size was 10% of the total number of samples. Gray images from the original dataset were discarded so as not be detrimental to the training of our models.

### Models Construction & Training

#### U-Net
In order to divide data evenly by the hue, first we calculate the hue value of each colorful image, after removing 581 gray-scale images, we sorted the images according to the hue value. Then we take 10% of the sorted set for validation, and the rest for training, both with similar color distributions.

To train the model, we need both gray images (input data) and corresponding colorful im- ages(ground truth). Then we converted the colorful images to the L a b color space. Gray images only contains the L value and the corresponding colorful image has L, a and b values. The goal is that our model can use L value to predict both a and b values then get reasonable results. Therefore, we trained the model with Adam’s optimizer and minimize MSE between the predicted and ground truth averaged over all pixels.

#### DCGAN


#### NoGAN


### Results

The predictions of U-Net, DCGAN and NoGAN and the ground truth of random selected samples are visualized in a figure to compare.

