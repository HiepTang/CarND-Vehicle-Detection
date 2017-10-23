# Vehicle Detection Project
The goal of this project is to write a software pipeline to detect vehicles on a video (start with the test_video.mp4 and later implement on full project_video.mp4).
The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradient (HOG) feature extraction.
* Apply a color transform, histogram of color and spatial binning of color.
* Prepare data and train a classifier.
* Implement a sliding window search technique and use it to search vehicles on test images.
* Implement pipeline and run on video stream.
* Combine with the advanced lane finding project.

Output:
* [Source Code](https://github.com/HiepTang/CarND-Vehicle-Detection/blob/master/Vehicle-Detection-Project.ipynb)
* [Output images](https://github.com/HiepTang/CarND-Vehicle-Detection/tree/master/output_images)
* [Test Video Output](test_video_out.mp4)
* [Project Video Output](project_video_out.mp4)
* [Project Video Lane Output](project_video_out_lane.mp4)

## Histogram of Oriented Gradients (HOG)
### Extract HOG features
Use the scikit-image hog function to extract the HOG features of image. The hog function output depends on some important parameters such as orient, pixels_per_cell and cells_per_block. Below is the get_hog_feature() function from Udacity lession and some expirements on these parameters in order to select the best one.
```python
from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
```
The output of hog features experiement is on the [output_images/hog](output_images/hog) folder. After try to explore the orient parameter values 7,9,11 with pix_per_cell=8 and cell_per_block=2, I choose the orient=9 because it gives the most clear shape of the car. Let's keep the orient=9 and cell_per_block=2 and try to some experiments on pix_per_cell parameter with values 6,8 and 10. From the result, I see the pix_per_cell=8 gives the most clear output. So I choose orient=9, pix_per_cell=8 and cell_per_block=2.
### HOG features and color space
In order to get the better result, I try to explore HOG features on the different color spaces. From the experiment result, I see there is no much different on HOG features between color spaces and it's good for classification. Please refer to the output images for different hog color space on the [output_images/hog/HLS](output_images/hog/HLS), [output_images/hog/HSV](output_images/hog/HSV), [output_images/hog/LUV](output_images/hog/LUV), [output_images/hog/YUV](output_images/hog/YUV) and [output_images/hog/YCrCb](output_images/hog/YCrCb) folders.
### Histogram of color
Let's try to explore the histogram of color on some different color spaces. From the result, I see the histogram on LUV color space give the best result with the clear different between the non vehicle and vehicle histogram. Please see the output images on [output_images/color_hist](output_images/color_hist) folder.
```python
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
```
### Spatial Binning of Color
You can perform spatial binning on an image and still retain enough information to help in finding vehicles. Let's try to explore the spatial binning of color on some different color spaces. I see the LUV and YUV color spaces give a good result to distinguish between vehicle and non vehicle images. Please see the output of this step on the [output_images/bin_spatial](output_images/bin_spatial) folder.
```python
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
```
### Combine together
Let's combine the hog features, spatial and color histogram together on the extract_features() function. From the above experiment result, I choose the parameters: orient=9, pix_per_cell=8, cell_per_block=2 and color_space=LUV. The remain parameters will be choosen on some next experiments.
```python
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        file_features = []
        feature_image = convert_color(image, color_space=color_space)
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
```

