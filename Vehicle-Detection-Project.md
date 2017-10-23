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
## Train a classifier
### Prepare training data
In this step, I implement the code to load all vehicle and non vehicle training images, extract hog features, histogram of color and spatial feature. Beside that I use the StandardScaler to normalize the data and random shuffle data to maxime the random on training.
```python
from sklearn.model_selection import train_test_split
from random import shuffle
import time
from sklearn.preprocessing import StandardScaler
# Prepare feature data function: extract hog features, histogram of color and spatial feature.
# Use StandardScaler to normalize data
# Shuffle the data to maximize the random on training
def prepare_feature_data(vehicle_imgs, non_vehicle_imgs, color_space='RGB', 
                        spatial_size=(16,16), hist_bins=16, 
                        orient=9, pix_per_cell=8, 
                        cell_per_block=2, 
                        hog_channel='ALL', spatial_feat=True, 
                        hist_feat=True, hog_feat=True):
    print('Color space:',color_space,'and',hog_channel,'Hog channel')
    print('Using orient of:',orient, 'and', pix_per_cell,'pix_per_cell','and',cell_per_block,'cell_per_block')
    print('Using spatial_size of:',spatial_size, 'and', hist_bins,'hist_bins')
    # Shuffle data
    shuffle(vehicle_imgs)
    shuffle(non_vehicle_imgs)
    # Load all vechile training images
    vehicle_features = extract_features(vehicle_imgs, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    non_vehicle_features = extract_features(non_vehicle_imgs, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Create an array stack of feature vectors
    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    
     # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    return X_train, X_test, y_train, y_test, X_scaler
```
### Train a classifier
The train classifier function use the linear SVC (Support Vector Classification). It is simple following the suggestion on the Udacity lession.
```python
from sklearn.svm import LinearSVC
def train_SVC(X_train, X_test, y_train, y_test):
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts    : ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    return svc
```
Train the support vector classifier (SVC) with the LUV color space on L channel, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32,32) and hist_bins=16. It gives 98.14% accuracy on test dataset and 100% correction on 10 random prediction.
```python
# Define parameters for feature extraction
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


X_train, X_test, y_train, y_test, X_scaler = prepare_feature_data(vehicle_imgs, non_vehicle_imgs, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
svc = train_SVC(X_train, X_test, y_train, y_test)
```
Train the support vector classifier (SVC) with the LUV color space on L channel, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32,32) and hist_bins=32. It gives 98.11% accuracy on test dataset and 100% correction on 10 random prediction.
```python
# Define parameters for feature extraction
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


X_train, X_test, y_train, y_test, X_scaler = prepare_feature_data(vehicle_imgs, non_vehicle_imgs, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
svc = train_SVC(X_train, X_test, y_train, y_test)
```
Train the support vector classifier (SVC) with the LUV color space on ALL channels, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(16,16) and hist_bins=16. It gives 98.31% accuracy on test dataset and 100% correction on 10 random prediction.
```python
# Define parameters for feature extraction
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


X_train, X_test, y_train, y_test, X_scaler = prepare_feature_data(vehicle_imgs, non_vehicle_imgs, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
svc = train_SVC(X_train, X_test, y_train, y_test)
```
## Implement sliding window search
### Same sliding window size search implementation
In this step, I implement some basic function use for sliding window search technique. It includes the draw_boxes() function use to draw the windows on an image, the slide_window() function use to calculate the windows need to search on an image, the single_image_feature() use to extract features on a single image and the search_windows() function use these functions in order to search vehicle on an image. All of these functions are copied from the Udacity lession.
I try to test these functions on the test_images in order to choose some remain paramters. Here are results:
* The SVC with the LUV color space on ALL channels give a very good result on accuracy with 98.42% correction however it gives a not good result on test images with some white car cannot detect.
* The SVC training with LUV color space on the L channel also give a good accuracy 98.17% and it can give a good result on test images with all cars are detected with some false detections. It takes about 0.60 seconds to detect vehicles on a test image.
Please view the output of this step on [output_images/test_images/same](output_images/test_images/same)
### Multi scale sliding window search
It's easy to regconize that the car appears on the image with different size based on their location. So in order to improve the search performance, I implement the slide window search with multiple scale.
```python
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
# y_start_stop=[400, image.shape[0]]
def slide_multi_scales_window(img, x_regions=([None, None],[None,None]), y_regions=([400, 528], [528, None]), 
                    xy_windows=([64, 64], [128,128]), xy_overlap=(0.5, 0.5)):
    # Initialize a list to append window positions to
    window_list = []
    for idx, y_region in enumerate(y_regions):
        x_region = x_regions[idx]
        if x_region[0] == None:
            x_region[0] = 0
        if x_region[1] == None:
            x_region[1] = img.shape[1]
        if y_region[0] == None:
            y_region[0] = 0
        if y_region[1] == None:
            y_region[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_region[1] - x_region[0]
        yspan = y_region[1] - y_region[0]
        xy_window = xy_windows[idx]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_region[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_region[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
    
    # Return the list of windows
    return window_list
debug = True
def find_vehicles_multi(image, svc, X_scaler, x_regions=([None, None],[None,None]), 
                  y_regions=([400, 528], [528, None]), xy_windows=([64, 64], [128,128]), 
                        xy_overlap=(0.5, 0.5), draw_windows=True):
    t=time.time()
    
    
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    windows = slide_multi_scales_window(img, x_regions=x_regions, y_regions=y_regions, 
                    xy_windows=xy_windows, xy_overlap=xy_overlap)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    t2 = time.time()
    if debug:
        print(round(t2-t, 2), 'Seconds to find vehicles...')
    if draw_windows:
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        return window_img, hot_windows
    else:
        return hot_windows
print('Done')
```
Please view the output of this step on [output_images/test_images/slide](output_images/test_images/slide) folder. It gives a better vehicle detection and improve performace a lot with only 0.30 seconds to detect vehicles on an image.
### Multiple detection and false positives
In order to combine the overlapping vehicle detections and avoid false positives, I apply the heat-map and threshold technique following the Udacity lession.
```python
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def filter_by_heatmap(img, box_list, threshold=1 ):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    return heatmap,labels
```
It gives me the better vehicle detection result on test images. Please view the [output_images/test_images/final](output_images\test_images\final) for more detail.
## Put all together - implement video pipeline
### Define a class to store vehicle detections history
The Vehicle_Detect_History class use to store the detection hot windows on video frames with the maximum size. It will be use for apply the heat map and threshold in order to give the better detection and eliminate false positives.
```python
class Vehicle_Detect_History:
    def __init__(self):
        self.hot_windows = []
        self.maximum = 15
        
    def add_hot_windows(self, windows):
        self.hot_windows.append(windows)
        if (len(self.hot_windows) > self.maximum):
            # remove the oldest one
            del self.hot_windows[0]
```
### Video pipeline implementation
```python
def add_heatmap(img, box_lists, threshold=1 ):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for box_list in box_lists:
        # Add heat to each box in box list
        heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    return heatmap,labels
 
 debug=False
def process_img(img):
    hot_windows = find_vehicles_multi(img, svc, X_scaler, x_regions=([None, None],[400,None],[400,None],[None,None]), 
                  y_regions=([400, 528], [400, 528], [400, 528], [528, 680]), 
                  xy_windows=([64, 64], [80,80], [96,96], [128,128]), xy_overlap=(0.5, 0.5), draw_windows=False)
    if (len(hot_windows) > 0):
        vdh.add_hot_windows(hot_windows)
    threshold = len(vdh.hot_windows)
    if threshold >= vdh.maximum:
        threshold = threshold - 3
    heatmap, labels = add_heatmap(img,vdh.hot_windows,threshold=threshold)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```
### Apply on the test video
The test video processing pipiline works well and give a good result. Please see more at [Test Video Output](test_video_out.mp4)
### Apply on the project video
The project video processing pipeline works so good with almost vehicles are detection. However, there are some need improvements with false positives and some difficult to detect the white car on some frames. Please see the result at [Project Video Output](project_video_out.mp4)
## Combine with the Advance Lane Finding project
After confident with the vehicle detection solution, I try to combine it with the Advanced Lane Finding solution. Please see the output at [Project Lane Output](project_video_out_lane.mp4).
## Disucssion
It is very wonderful project. I learnt a lot from this project. My solution gives a good result however I think there are some improvements that I'd like to do in the future:
* It has some difficults to detect the white car and make some false positives. It would be better if I try to increase the training data such as Udacity data.
* I'd like to explore more on some other machine learning algorithm such as Decision Trees.
* I think the deep learing approach will give a better result.
