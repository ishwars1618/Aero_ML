# Detecting landmarks of aircraft in the Aerofly RC10 Simulator

In this project, I experimented with different convolutional neural network architectures to extract landmark locations from images of a remote control F16 aircraft.

Here is an example.

![](example.png)
**Nose** is the nose of the aircraft, **L** is the left wingtip, **R** is the right wingtip, **T** is the tail point, and **V** is the tip of the vertical stabilizer.

The folder `plane_data ` contains images and their respective labeled points in CSV format. The folder `outp` visualizes these labellings, containing the original image and the annotated image. This labeling was performed on 900 frames in a 20-minute screen-recorded video, with the help of `label_video_multipoint.py` that enables the user to pan through a video frame-by-frame and click points on landmarks so that their locations are saved in CSV format.

This work uses heatmap regression, in which a photo of the aircraft is taken as an input and the output is a set of 5 heatmaps -- each heatmap is a likelihood distribution on where the landmark is in the image, and there are 5 heatmaps since there are 5 landmarks to detect.

I've experimented with different neural network architectures (such as a vanilla CNN, and the U-Net) as well as additional techniques like Dropout, Batch Normalization, and the activation functions used at the outputs of the model. The Jupyter Notebook `plane_landmarks.ipynb` has the current working version of the model architecture + training & inference flow, which is successful to some degree but can be improved much more. Please see an example of the model performing live inference at `live-inference.mov` using the model trained in `plane_landmarks.ipynb`.

#Improvements

The training data can be augmented at least 10x using transformations like rotations, scaling, and shears, as well as linear color-map transformations (transformations that map `(r, g, b)` to `A*(r, g, b)` where `A` is a 3x3 matrix with potentially rank 3, 2, or 1) so that the model recognizes color patterns rather than colors themselves on the aircraft (like red, white, black, and blue). In addition, the blue background behind the aircraft can be changed for other arbitrary backgrounds, which can be drawn from a separate image database.

Instead of using heatmap regression, a coordinate regression model can be made, which outputs the values of the coordinates of each landmark (5*2 = 10 floating point values). This approach is known to be more robust.

To eliminate the bottleneck of manually labelling data, some works have introduced unsupervised learning methods that segment images without any point data, which I plan to explore more in depth.

