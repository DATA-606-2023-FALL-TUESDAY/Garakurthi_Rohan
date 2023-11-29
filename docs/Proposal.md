# 1. Title and Author

- Vehicle Type Detection for Intelligent Transportation Systems & Traffic Management
- Rohan Garakurthi
- https://github.com/Rohan198/UMBC-DATA606-FALL2023-TUESDAY
- https://www.linkedin.com/feed/?trk=guest_homepage-basic_google-one-tap-submit
- https://docs.google.com/presentation/d/1BZiTZgs6JVvtrBKOSgS4Ys6amEEv9bCi/edit?usp=drive_link&ouid=112142881428497726904&rtpof=true&sd=true
    
# 2. Background
In today's rapidly evolving world, intelligent transportation systems are essential for the smooth and efficient functioning of our cities and economies. As vehicles of all shapes and sizes traverse our road networks, it becomes increasingly crucial to have automated systems that can classify and identify these vehicles accurately. 

This project "Vehicle Type Detection" aims to provide a versatile solution for a multitude of applications by leveraging the power of deep learning and computer vision to analyze images and automatically categorize vehicles into distinct types. 

From improving traffic management and surveillance to enhancing security and supporting autonomous vehicles, the implications of accurate vehicle type detection are vast. We explore the intricacies of identifying vehicles from images, ultimately contributing to smarter, safer, and more efficient transportation systems.

Some practical application where vehicle type detection can be of use:
●	Road Safety, Traffic Management and Surveillance
●	Smart Parking Solutions, Toll Collection and Access Control
●	Vehicle Insurance Risk Assessment

Research Question(s):
1.	How accurately simple CNN models with less layers can classify this data?
2.	What parts of images are concentrated by the models during classification?
3.	How pretrained models based on deep CNN layers compare in classifying vehicle types?
4.	How different data augmentation techniques can impact the model performance?
5.	How accurately can different model(s) classify the data?


# 3. Data 


Link: https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition

●	Number of Images: 400
●	Number of Vehicles types/classes : 4
●	Datasize: =~ 200 MB

The dataset comprises a diverse collection of images for four distinct classes (100 of each class): Car, Truck, Bus, and Motorcycle. 

More data collection: While the size of this dataset is small, based on initial results from the model training & evaluation part, I plan to extend the size of the data set with help of web scraping or bringing in the data from other datasets (like Open Images dataset).

# 4. Exploratory Data Analysis (EDA)

- Used matplotlib to load and view the data samples for each class.
- Used tensorflow image module to augment the input data by various techniques such as decolorizing, sharpening, rotating, resizing & cropping.
- Visualized the feature maps from the CNN layers (after training) to determine based on which parts of images the models are predicting the vehicle types.

## Image Data Augmentation

Image augmentation is a technique used to increase the diversity of training examples for computer vision tasks, especially when data size is small. These techniques involve applying various transformations to input images to create new variations of the data without collecting additional real-world samples. This step helps to improve model generalization and reduces overfitting. Below are the explanations of some common image data augmentation techniques/methods:

1. Rotation:

    ○ Rotate the image by a certain angle (e.g., 90 degrees, 180 degrees) to introduce variability in object orientations, making the model more robust to rotated objects.

2. Vertical or Horizontal Flipping:

    ○ Flip the image horizontally or vertically, so that the model learns invariant features from left-to-right or top-to-bottom orientations.

3. Random Cropping and Resize:

    ○ Crop a portion of the image and resize it to the original size. This helps the model to focus on different regions of the image while simulating different viewpoints or zoom levels.
4. Scaling and Zooming:

    ○ Randomly scale the image up or down, providing variations in the size of objects, helping the model generalize to different object sizes.
5. Noise Addition:

    ○ Add random noise (e.g. Gaussian noise) to the image to simulate noise in real-world images, improving the model's ability to handle noisy data.
6. Blur:

    ○ Apply various blur filters (e.g., Gaussian blur, motion blur) to the image to reduce sharpness, making the model more robust to images with different levels of clarity.
7. Brightness/Contrast/Saturation Adjustment:
    
    ○ Adjust the image by adding or subtracting a random value which helps the model adapt to varying lighting conditions, color intensity.

## Image data augmentation using tf layers
As discussed above, image augmentation techniques can be used individually or in combination to create a wide range of augmented images from a single original image while training a computer vision model. This technique can improve the model's ability to generalize to different conditions and variations in the input data. For this project, TensorFlow image augmentation methods are used to perform various transformations on an input image. Below is the explanation of methods used.

1. tf.image.flip_left_right(image):

       ○ This method horizontally flips (mirrors) the input image from left to right.
2. tf.image.rgb_to_grayscale(image):

       ○ This method converts a color image (RGB) into grayscale, resulting in a single-channel image. This method can be useful for certain    tasks such as edge detection or reducing the computational cost of processing.
3. tf.image.central_crop(image, central_fraction=0.5):

       ○ This method crops the central portion of the input image based on the specified central fraction. In the code, a central fraction of 0.5 is used to keep the central 50% of the image.

4. tf.image.rot90(image):

       ○ Function: This method rotates the input image by 90 degrees counterclockwise.
5. tf.image.adjust_saturation(image, 3):

       ○ Function: This method adjusts the saturation of the input image. In the code, it increases the saturation by a factor of 3.
6. tf.image.adjust_brightness(image, 0.4):

       ○ Function: This method adjusts the brightness of the input image by adding a specified brightness value (0.4 in the code).
   
# 5. Model Training 

- Used matplotlib to load and view the data samples for each class. Split the dataset into training and validation datasets for evaluating the model.
- Developed deep learning architectures based on CNN’s to classify the vehicle type images.
- Experimented with different weight initialization techniques & normalization layers.
- Used pretrained models such as Resnet, Efficient Net, MobileNet & finetune the classification layer for the accurate predictions.
- Visualized the model's loss and predictions alongside actual/expected predictions.

## Baseline CNN Model:

TensorFlow and Keras are used to build a simple baseline CNN model using deep neural networks. The convolutional layers are used to extract hierarchical features from images, and the fully connected layers provide the final classification output. This model can be trained on labeled image datasets to learn to classify images into different classes.

## Data Augmentation:

○ A sequential data_augmentation model is defined to preprocess input images. It applies the following transformations:
■ Rescaling: Scales pixel values to the range [0, 1].
■ Resizing: Resizes images to a target size of 224x224 pixels.
■ Random Horizontal and Vertical Flipping: Randomly flips images horizontally and vertically, introducing variations in object orientations.
■ Random Rotation: Applies random rotations of up to 0.2 radians to simulate different object orientations.

## Model Architecture:

○ The main model is a sequential model that consists of the following layers:
■ Data augmentation layer: As described in the above section, a number of image augmentation layers are used.
■ Convolutional Layer: A 2D convolutional layer with 16 filters, a kernel size of 3x3, 'same' padding, and ReLU activation. This layer extracts features from the input images.
■ Max Pooling Layer: Performs max-pooling to reduce the spatial dimensions of the feature maps.
■ Flatten Layer: Flattens the output from the previous layers into a 1D vector.
■ Dense Layer (64 units): A fully connected layer with 64 units and ReLU activation.
■ Dense Layer (output): The final fully connected layer with the number of units equal to the number of classes (4 in our case). This layer produces the model's output logits.

## Input Shape & Model Compilation:

○ The model's input shape is specified as (None, 224, 224, 3), indicating that it expects input images with a size of 224x224 pixels and 3 color channels (RGB).

○ The model is compiled with the following settings:

■ Loss Function: Sparse Categorical Cross-Entropy, suitable for multi-class classification tasks.

■ Metrics: Accuracy is monitored during training to evaluate model performance.

■ Optimizer: Adam optimizer

## Next Steps: Improve dataset quality by extending data & classes

● Add more images into 4 existing classes of datasets to improve the quality of the dataset and the machine learning model.

● Extend classes available by finding more data, to add other vehicle types like airplane & train.

● Add some random non-vehicle images to a class named "non-vehicle" so that models know when a vehicle is not present in the given input.

● Compare the performance of different pretrained models.

# 6. Application of the Trained Models

- I plan to use streamlit to build a web app to upload vehicle images and classify the vehicle type.
 
# 7. Conclusion

- Summarize your work and its potetial application
- Point out the limitations of your work
- Lessons learned 
- Talk about future research direction

## Additional datasets under consideration:

● https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset

● https://www.kaggle.com/datasets/maciejgronczynski/vehicle-classification-dataset

● https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set

# 8. References 

- Data Augmentation using Tensorflow, https://www.tensorflow.org/tutorials/images/data_augmentation
- Image plotting with matplotlib, https://matplotlib.org/stable/tutorials/images.html
