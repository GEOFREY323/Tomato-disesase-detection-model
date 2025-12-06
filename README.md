# Tomato Disease Detection Project Report

This report details the development of a Convolutional Neural Network (CNN) model for detecting diseases in tomato leaves and a Streamlit application for deploying the model.

## Project Setup and Data Handling

The project commenced with the installation of essential libraries, including `opencv-python` for image processing, `tensorflow` for neural network development, and `matplotlib` for data visualization. Standard libraries like `numpy` for numerical operations, `os` for file system interactions, and `sklearn` for data splitting were also incorporated.

A function named `load_data` was implemented to manage the dataset. This function reads images from specified directories, treating each subdirectory as a distinct class. Images are resized to a uniform 128x128 pixels using `cv2.resize`. The image data is stored in a NumPy array, and corresponding numerical labels, derived from the alphabetical order of class names, are stored separately. A `class_map` dictionary was created to facilitate the mapping between class names and their numerical indices.

Data preprocessing involved normalizing the pixel values of the images by dividing them by 255.0, scaling them to the range [0, 1). The raw numerical labels were converted into a one-hot encoded format using `to_categorical`, a requirement for the chosen loss function during model training.

To address the possibility of encountering images that do not represent tomato leaves, an 'Unknown' class was introduced. The code calculates the average number of images per existing class to determine a suitable target count for the 'Unknown' class, aiming for a balanced dataset. A directory for this class was created. Example non-tomato leaf images were intended to be downloaded into this directory and subsequently resized to the standard image size.

## Model Architecture and Training

A Convolutional Neural Network (CNN) model was constructed using the `Sequential` API from TensorFlow/Keras. The architecture comprises several layers:
- The initial layer is a `Conv2D` layer with 32 filters, a (3,3) kernel size, ReLU activation, and an input shape of (128, 128, 3).
- Following this, a `MaxPooling2D` layer with a pool size of (2,2) was used to reduce spatial dimensions.
- Two more pairs of `Conv2D` (with 64 and 128 filters, respectively, both with (3,3) kernels and ReLU activation) and `MaxPooling2D` (with (2,2) pool size) layers were added.
- A `Flatten` layer was included to convert the output of the convolutional layers into a 1D vector.
- Two `Dense` layers were used: the first with 128 units and ReLU activation, and the final output layer with `len(class_map)` units (11 in this case, covering the 10 disease classes and 'Unknown') and a softmax activation for class probability output.

The model was compiled using the `adam` optimizer and the `categorical_crossentropy` loss function, which is suitable for multi-class classification with one-hot encoded labels. The `accuracy` metric was used to evaluate performance.

The model underwent training for 10 epochs using the training dataset (`X_train`, `y_train`). Validation was performed at the end of each epoch using a separate validation dataset (`X_val`, `y_val`) to monitor for overfitting.

Upon completion of training, the model's performance was evaluated on an unseen test dataset (`X_test`, `y_test`). The evaluation indicated a test loss of approximately 1.3540 and a test accuracy of approximately 0.7310 (73.10%).

Analysis of the training history plots revealed that while training accuracy increased and training loss decreased consistently, the validation loss began to increase or fluctuate significantly after a few epochs. This divergence between the training and validation curves suggests that the model may be overfitting the training data, indicating a potential reduction in its ability to generalize to new, unseen images. Strategies to mitigate overfitting, such as incorporating dropout layers, utilizing data augmentation, or implementing early stopping, could be considered for future improvements.

## Streamlit Application Deployment

A Streamlit application (`app.py`) was developed to provide a user-friendly interface for the trained tomato disease detection model.

The application initiates by loading the pre-trained Keras model from the specified path. A `class_map` and its reverse (`reverse_map`) were defined to handle the mapping between class names and numerical indices.

A function named `preprocess_image` was created within the app to prepare images for model input. This function handles different color formats, resizes images to 128x128 pixels, normalizes pixel values, and adds a batch dimension.

The core of the application's functionality lies in the `predict` function. This function takes an image as input and performs a crucial initial check for non-leaf images based on green pigment content. It converts the image to the HSV color space, creates a mask for green pixels, and calculates the percentage of green pixels. If this percentage falls below a defined threshold (10%), the function immediately identifies the image as "Unknown (Not a Leaf)". If sufficient green pigment is detected, the image is preprocessed and fed into the loaded model for prediction. The function then determines the predicted class and confidence. A confidence threshold (0.6) is applied, and if the predicted class is "Unknown" or the confidence is below this threshold, the output is "Unknown". Otherwise, the specific predicted disease class and confidence are returned.

The Streamlit user interface was configured with a page title and centered layout. It displays the main title of the app and the calculated test accuracy. The training and validation history plots are also presented. The UI includes a sidebar allowing users to choose between uploading an image or taking a photo. The selected image is displayed, and a button triggers the prediction process. The predicted class label and confidence are then shown to the user.

---

Done by Gam Geofrey Ankinimbom, an intern at Traitz Tech, as part of my final internship project.