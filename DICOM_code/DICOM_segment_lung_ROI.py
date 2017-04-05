import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob

working_path = "/Volumes/G-DRIVE mobile/KaggleData/output/"

file_list = glob(working_path + "images_*.npy")

for img_file in file_list:
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    imgs_to_process = np.load(img_file).astype(np.float64)  # this is an array
    print "on image", img_file
    for i in range(len(imgs_to_process)):  # len is the number of slices, which is variable
        print i
        img = imgs_to_process[i]  # this is a single image slice, an array with heightXwidth
        # Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)  # calculates the standard deviation along the specified axis (which is an array here??)
        img = img - mean
        img = img / std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400, 100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        img[img == max] = mean
        img[img == min] = mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        # prod gives you the product of the array elements
        # reshape gives a new shape to the array without changing its data (so we turn the middle into a vector)
        # fit computes the clustering on the given data
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region
        # engulf the vessels and incursions into the lung cavity by
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions
        #  with a bbox that is too large in either dimension
        #  Also, the lungs are generally far away from the top
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity.
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        mask = np.ndarray([512, 512], dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask
        #
        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
        imgs_to_process[i] = mask
    np.save(img_file.replace("images", "lungmask"), imgs_to_process)

#
#    Here we're applying the masks and cropping and resizing the image
#


file_list = glob(working_path + "lungmask_*.npy")
for fname in file_list:
    print "working on file ", fname
    patient_out_images = []
    patientID = fname.replace(".npy", "")
    patientID = patientID.replace("lungmask_", "")
    patientID = patientID.replace(working_path, "")
    print "patient ID: " + patientID
    imgs_to_process = np.load(fname.replace("lungmask", "images"))  # replace the WHOLE image with just the lungs masked out
    masks = np.load(fname)
    for i in range(len(imgs_to_process)):
        print i
        mask = masks[i]  # this is a slice of the lung mask
        img = imgs_to_process[i]  # this is a slice of the original image
        img = mask * img  # apply lung mask
        #
        # renormalizing the masked image (in the mask region)
        #
        new_mean = np.mean(img[mask > 0])
        new_std = np.std(img[mask > 0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        old_min = np.min(img)  # background color
        img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
        img = img - new_mean
        img = img / new_std
        # make image bounding box  (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col - min_col
        height = max_row - min_row
        if width > height:
            max_row = min_row + width
        else:
            max_col = min_col + height
        #
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        #
        img = img[min_row:max_row, min_col:max_col]
        mask = mask[min_row:max_row, min_col:max_col]
        if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no good regions
            pass
        else:
            # moving range to -1 to 1 to accomodate the resize function
            mean = np.mean(img)
            img = img - mean
            min = np.min(img)
            max = np.max(img)
            img = img / (max - min)
            new_img = resize(img, [512, 512])
            patient_out_images.append(new_img)

    num_patient_images = len(patient_out_images)
    print "number of patient images: " + str(num_patient_images)
    #
    #  Writing out images and masks as 1 channel arrays for input into network
    #
    final_patient_images = np.ndarray([num_patient_images, 1, 512, 512], dtype=np.float32)
    for i in range(num_patient_images):
        final_patient_images[i,0] = patient_out_images[i]

    np.save(working_path + "ROI_%s.npy" % patientID, final_patient_images)


# # For checking the ROI isolating
# import matplotlib.pyplot as plt
#
# imgs = np.load(working_path + 'images_0ddeb08e9c97227853422bd71a2a695e.npy')
# lungmask = np.load(working_path + 'lungmask_0ddeb08e9c97227853422bd71a2a695e.npy')
# roi = np.load(working_path + 'ROI_0ddeb08e9c97227853422bd71a2a695e.npy')
# print len(roi)
# for i in range(len(imgs)):
#     if i % 10 == 0:
#         print "image %d" % i
#         fig, ax = plt.subplots(2, 2, figsize=[8, 8])
#         ax[0, 0].imshow(imgs[i], cmap='gray')
#         ax[0, 1].imshow(lungmask[i], cmap='gray')
#         ax[1, 0].imshow(imgs[i] * lungmask[i], cmap='gray')
#         ax[1, 1].imshow(roi[i, 0], cmap='gray')
#         plt.show()
#         raw_input("hit enter to cont : ")