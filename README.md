# Image-Colorization

This is the final project for the [DD2424 - Deep Learning in Data Science](https://www.kth.se/student/kurser/kurs/DD2424?l=en) course of KTH Royal Institute of Technology.

Supervisor: [Josephine Sullivan](https://www.csc.kth.se/~sullivan/), KTH Royal Institute of Technology

## Task
Explore and compare different deep learning models for image colorization, three representative methods we choosed are U-Net, DCGAN and NoGAN.

## Methodology & Procedure

### Dataset

CIFAR-10 dataset is chosen for this project, which contains 50000 32\*32 images that can be classified to 10 classes.

In our project, two train-validation data subsets from CIFAR-10 are tested. The first one consists of a number of randomly selected samples. The second subset is selected from a sorted set by mean hue value of all the pixels. In this way, generating sets according to a specific proportion so that they all contain similar color distributions and can effectively avoid biasing to any specific color. The chosen validation set size was 10% of the total number of samples. Gray images from the original dataset were discarded so as not be detrimental to the training of our models.

### Models Construction & Training

#### U-Net

U-Net is a fully convolutional network model. The contractive path consists of 4 × 4 convolution layers with stride 2 for down-sampling, each followed by batch normalization and Leaky-ReLU activation function with the slope of 0.2. The number of channels is doubled after each step. Each unit in the expansive path consists of a 4 × 4 transposed convolutional layer with stride 2 for up-sampling, concatenation with the activation map of the mirroring layer in the contracting path, followed by batch normalization and ReLU activation function. The last layer of the network is a 1 × 1 convolution which is equivalent to cross-channel parametric pooling layer. We use the tanh function for the last layer.

<div align=center>
<img src=https://github.com/MUYang99/Tensorflow_Image-Colorization-with-Deep-Learning/blob/main/img/U-NET.png/>
</div>

In order to divide data evenly by the hue, first we calculate the hue value of each colorful image, after removing 581 gray-scale images, we sorted the images according to the hue value. Then we take 10% of the sorted set for validation, and the rest for training, both with similar color distributions.
 
To train the model, we need both gray images (input data) and corresponding colorful images(ground truth). Then we converted the colorful images to the L a b color space. Gray images only contains the L value and the corresponding colorful image has L, a and b values. The goal is that our model can use L value to predict both a and b values then get reasonable results. Therefore, we trained the model with Adam’s optimizer and minimize MSE between the predicted and ground truth averaged over all pixels.

#### DCGAN
The basic task for our GAN is to add three channels (RGB) with relevant intensities of each color channel. Hence, to address this problem, we use a special flavor of GAN called Conditional DCGAN which accepts gray scale images (with one intensity channel) as input. The discriminator input is also changed to be compatible with the conditional DCGAN. Our final cost functions are as follows then:

<div align=center>
<img src=https://github.com/MUYang99/Tensorflow_Image-Colorization-with-Deep-Learning/blob/main/img/formula.png/>
</div>

#### NoGAN
Basic steps for NoGAN are as follows: First train the generator in a conventional way by itself with just the feature loss. Next, generate images from that, and train the critic for distinguishing between those outputs and real images as a basic binary classifier. Finally, train the generator and critic together in a GAN setting.

### Results

The predictions of U-Net, DCGAN and NoGAN and the ground truth of random selected samples are visualized in a figure to compare.


