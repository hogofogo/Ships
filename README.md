# Ships

This projects examined a database of ship images from Kaggle: https://www.kaggle.com/rhammell/ships-in-satellite-imagery.

The objective was to detect whether or not a ship is present in this image, i.e. it's a simple binary classification. As provided, the data set was not challenging (see example of Kaggle images), but I complicated the task. 

The initial set contains 2800 images, of which 700 are labeled as '1' and the rest as '0', including ~1/4 of the images that had been misclassified at origin as not ships, whereas they clearly are. For example, the 'good' images were labeled as '1' in cases where images were those of full ships, including bow and stern, and traversing the center of the image. The mis-centered images, or the incomplete ones, had '0' labels. 

The data labels are part of the file names; therefore I updated the file names for such misclassified images to correctly reflect their ship/no ship status. The exceptions included cases where I was not sure if I am indeed looking at a ship, or when only a very small part of the ship is visible in the corner of the image, with neither a bow or a stern. As a result of the changes, '1' and '0' labels were approximately balanced and correctly reflecting the nature of the images.

I started with a small network with only one convolutional layer which had been previously successful at correctly getting >90 percent classification on the testing set before the complications I made to the training set (see above). With the simple model, I could not get an acceptable performance on either the training or testing set and made changes to the model, adding two convolutional layers for the total of 3 followed by 1 fully connected layer with a sigmoid activation.

With a modest amount of training I achieved >91 percent correct classification on the training set which can probably be further improved without a high risk of overfitting.

File Ships_v1 contains everything required for data prep, the trainable model, and the function for examining the image for the correctness of its classification classify_random_image().

Model with trained weights can be loaded by running:
shipModel = load_model('/Users/vlad/Projects/ships-in-satellite-imagery/shipModel.h5')