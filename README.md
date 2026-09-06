# Detecting landmarks of aircraft in the Aerofly RC10 Simulator

In this project, I experimented with different convolutional neural network architectures to extract landmark locations from images of a remote control F16 aircraft in a flight simulator.

Here is an example.

![](ignore/example.png)
**Nose** is the nose of the aircraft, **L** is the left wingtip, **R** is the right wingtip, **T** is the tail point, and **V** is the tip of the vertical stabilizer.

This work uses heatmap regression, in which a photo of the aircraft is taken as an input and the output is a set of 5 heatmaps -- each heatmap is a likelihood distribution for where the corresponding landmark is in the image. Since there are 5 landmarks to detect (nose tip, left wing tip, right wing tip, rear end, vertical stabilizer tip), the model predicts 5 heatmaps.

I've experimented with different neural network architectures (such as a vanilla CNN and the U-Net). The Jupyter Notebook `plane_landmarks.ipynb` has the current working version of the model architecture + training & inference flow. Please see an example of the model performing live inference at `live-inference.mov` using the model trained in `plane_landmarks.ipynb`.

# Dataset

The folder `plane_data` contains images and their respective labeled points in CSV format. The folder `outp` visualizes these labelings, containing the original image and the annotated image. I labeled 900 frames in a 20-minute screen-recorded video, with the help of `label_video_multipoint.py` that enables the user to pan through a video frame-by-frame and click points on landmarks so that their locations are saved in CSV format.

The model is trained on these input+output image pairs:
* input: an image of dimension `(3, model_image_size, model_image_size)`, with values in the range (-1, 1).
* outputs: an image of dimension `(5, model_image_size, model_image_size)`, where each of the 5 images is a probability map whose values sum to 1. The target for each probability map is $N(\mu, \sigma^2)$ where $\mu$ is the pixel coordinates of the landmark, and $\sigma < 5$ pixels. See `coord_to_heatmap()` for implementation details.

# Data Augmentation

Because the pixels in images of the training data comprise a limited set of colors (namely blue, white, and red), I perform data augmentation color-wise using the `RandomColorMatrix` class, which performs linear transformations over the (R, G, B) space with restricted spectral norm. 

I also use the `RandomAffineWithPoints` class to transform images through a composition of rotations, scaling, translations, and shears, and a `RandomAdditiveNoise` to add noise to the images. 

# Model architectures

## U-Net

## Transformer

# Future work

* Create a coordinate regression model (which directly predicts the coordinates of the landmarks instead of inferring them from probability heatmaps), for comparison with the existing approaches.
* To eliminate the bottleneck of manually labeling data, investigate unsupervised learning methods for segmenting images, with the goal of predicting landmark locations for other kinds of aircraft.
* To improve the accuracy of the model for images where the background has nontrivial textures (e.g. terrain, clouds), fill the background of airplane dataset images with randomly chosen images from ImageNet dataset when training.
* Train a Recurrent Neural Network to predict the present position, altitude, & attitude of the aircraft, given the sequence of previous time-varying remote control inputs, to essentially replace the simulator; if there's no way to programmatically extract the raw position, altitude, & attitude information directly from the simulator, this would require using screen recordings of the reported position & altitude numbers and recordings of the plane itself, on which the landmark model would be applied to derive the attitude.
