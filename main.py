
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Dataset size
size = range(0, 9)

#Function to read images from a folder name with given range
def readDataset(folderName, setSize):
    images = []

    for i in setSize:
        if i != 2: #For some reason 2_rendered.png doesn't exist in the dataset...
            statement = str(i) + "_rendered.png"
            images.append((cv2.cvtColor(cv2.imread(folderName + "\\" + statement), cv2.COLOR_BGR2RGB), statement))
    return images

def saveDataset(folderName, images):
    for image in images:
        cv2.imwrite(folderName + "\\" + image[1], image[0])

def getHSVDataset(images):
    HSVImages = []

    for i in size:
        HSVImages.append((cv2.cvtColor(images[i][0], cv2.COLOR_RGB2HSV), images[i][1].replace(".png", "_HSV.png")))

    return HSVImages

def showDataset(images):

    for i in size:
        cv2.imshow(images[i][1], images[i][0])

    cv2.waitKey(0)

def threeChannelHistogram(images):

    histogram = []
    for i in size:
        c1_hist = cv2.calcHist([images[i][0]], [0], None, [256], [0, 256])
        c2_hist = cv2.calcHist([images[i][0]], [1], None, [256], [0, 256])
        c3_hist = cv2.calcHist([images[i][0]], [2], None, [256], [0, 256])
        all_channels_hist = (c1_hist, c2_hist, c3_hist)
        histogram.append(all_channels_hist)

    return histogram

def plot3ChannelHistogram(images, channel):
    histograms = threeChannelHistogram(images)
    for i, h in enumerate(histograms):
        for j, col in enumerate(channel):
            plt.figure()
            plt.plot(h[j])
            plt.title("image: " + str(i + 1) + " channel: " + col + " histogram")
            plt.xlabel("pixel intensity")
            plt.ylabel("frequency")
            plt.xlim([0, 256])
            plt.grid(True)
            plt.show()

#Segmentation via binary thresholding, apply to greyscale then set all pixes above threshold to 255(white),by applying THRESH_BINARY_INV
#We get images where the hands are white.

def threeChannelSegmentation(threshold, images):
    segmented_images = []
    for i in size:
        image = images[i][0]
        c1, c2, c3 = cv2.split(image)
        mask = (
            (c1 > threshold[0][0]) & (c1 < threshold[0][1]) &
            (c2 > threshold[1][0]) & (c2 < threshold[1][1]) &
            (c3 > threshold[2][0]) & (c3 < threshold[2][1])
        )
        #Apply mask to a black image with the same size and proportions
        segmented = np.zeros_like(image)
        #Foreground object (hand)
        segmented[mask] = [0, 0, 0]
        segmented[~mask] = [255, 255, 255]
        segmented_images.append((segmented, images[i][1].replace(".png", "_segmented.png")))
    return segmented_images



if __name__ == "__main__":
    #Get images from the Hand Dataset folder, with given range

    hands = readDataset("HandDataset", range(0, 10))
    #hands => Array of tuples. Tuple -> (image data, image_name)

    #--------------------
    #RGB SEGMENTATION
    #--------------------

    RGB = ('r', 'g', 'b')
    #Get RGB histogram for each image
    plot3ChannelHistogram(hands, RGB)
    #Based on the RGB histograms we can generalise all channels 150 - 255
    showDataset(threeChannelSegmentation(((150, 255), (150, 255), (150, 255)), hands))
    saveDataset("RGBSegmentedDataset",threeChannelSegmentation(((150, 255), (150, 255), (150, 255)), hands))

    #---------------------
    #HSV SEGMENTATION
    #---------------------

    HSV = ('h', 's', 'v')
    plot3ChannelHistogram(getHSVDataset(hands), HSV)
    #Based on the HSV histogram we can generalise the threshold such as: 150 hue, 160 saturation, 220 value
    showDataset(threeChannelSegmentation(((0, 120), (0, 30), (150, 190)), getHSVDataset(hands)))
    saveDataset("HSVSegmentedDataset", threeChannelSegmentation(((0, 120), (0, 30), (150, 190)), getHSVDataset(hands)))
    #RGB segmentation is much easier as binary thresholding works just fine for simple skin colour imagery however we get
    #a much clearer result with using HSV


