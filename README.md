                                       Color - classification

The main goal of this project is to evaluate the capability of Convolutional Neural Networks (CNNs) in detecting and classifying monochrome images based on different colors. It aims to investigate whether a CNN can accurately classify images according to various colors with high precision. Additionally, the project evaluates the network's performance on noisy data to understand how robust it is against noise and whether it can maintain its performance under such conditions.

The project involves training a neural network to classify monochrome images of different colors such as blue, green, red, yellow, black, white, etc. The Python includes various steps, such as dataset preparation, model building, training, evaluation, and testing. Here's an explanation of each major step:
1. Dataset Preparation:
 - The Python starts by importing necessary libraries and packages, including TensorFlow, Keras, NumPy, Matplotlib, and PIL (Python Imaging Library).
- It uses the Kaggle API to download a color classification dataset.
 - The downloaded ZIP file is then extracted to a specified directory.
 - Unwanted directories from the extracted dataset are deleted to clean up the data.
 2. Dataset Exploration:
 - The Python explores the dataset by counting the number of images and displaying examples of different colors.
3. Data Preprocessing:
 - The dataset is divided into training and validation sets using the `image_dataset_from_directory` function.
 - Data augmentation techniques are applied to the training dataset, including random horizontal flipping, rotation, and zooming.
4. Model Building:
- A convolutional neural network (CNN) model is defined using Keras Sequential API.
- The model consists of several layers, including data augmentation, rescaling, convolutional layers, max-pooling layers, dropout, flatten, and dense layers.
 - The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric.
5. Model Training:
- The model is trained using the `fit` function with the training and validation datasets.
- Training history is stored, including accuracy and loss values over epochs.
6. Model Evaluation:
 - The accuracy and loss curves are plotted using Matplotlib to visualize the model's performance during training.
 7. Testing:
- A separate test dataset is loaded for evaluating the model's accuracy on unseen data.
 - The model's accuracy on the test dataset is printed and stored in a variable.
 8. Adding Noise:
 - A noise factor is defined to introduce noise to the images
. - A function is defined to add noise to images in a batch.
- The test dataset is modified to include noisy images using the defined noise factor.
9. Noisy Dataset Evaluation:
- The model's accuracy is evaluated on the noisy test dataset.
- The accuracy of the model with noise is printed.
 10. Network Tolerance:
 - The Python explores the effect of different noise factors on the model's accuracy.
- The Python iterates through noise factors, applies noise to the test dataset, and evaluates the model's accuracy.
- If the accuracy falls below a specified threshold, the training loop is stopped.
 11. Final Testing with Network Tolerance:
 - The Python selects the noise factor that led to the model's accuracy falling below the tolerance threshold.
- The test dataset is modified with the chosen noise factor
- The final model's accuracy is evaluated and plotted on the noisy test images. Overall, this Python demonstrates the process of training a neural network for color classification, assessing its performance on noisy data, and exploring the network's tolerance to noise. It covers various aspects of deep learning, including data preparation, model building, training, evaluation, and testing, making it a comprehensive example project for color classification. 


Since in the downloaded file, the testimg folder was next to the colors folder and there were unlabeled test photos in that folder, first I deleted that folder in the training data and manually labeled the folder myself and uploaded it again.  and placed it in /content/testimg directory 


Testimg file link for use in test dataset:
https://drive.google.com/file/d/1Si2NvoEFdZ0JDD-JMWAUO6Au9sMZvSsv/view?usp=sharing  

