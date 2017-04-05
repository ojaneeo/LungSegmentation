import os
import numpy as np
import SimpleITK as sitk

output_path = "/Volumes/G-DRIVE mobile/KaggleData/output"

# Get the list of directories (dirs) = patients
baseRoot, dirs, files = next(os.walk('/Volumes/G-DRIVE mobile/KaggleData/testSlices'))

# For each patient, consolidate the files (slices) into a single npy array
for patient in dirs:
    print patient
    patient_path = baseRoot + "/" + patient

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(patient_path)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    img = sitk.GetArrayFromImage(image)

    np.save(os.path.join(output_path, "images_%s.npy" % patient), img)

# # For viewing .npy
# import matplotlib.pyplot as plt
# imgs = np.load(output_path+'/images2_0ddeb08e9c97227853422bd71a2a695e.npy')
# size = imgs.size
# print size
# size = len(imgs)
# print ("Length of file: " + str(size))
# for i in range(size):
#     print ("image %d" % i)
#     fig,ax = plt.subplots(2,2,figsize=[8,8])
#     ax[0,0].imshow(imgs[i],cmap='gray')
#     ax[0,1].imshow(imgs[(i+1)%size],cmap='gray')
#     ax[1,0].imshow(imgs[(i+2)%size],cmap='gray')
#     plt.show()
#     raw_input("hit enter to cont : ")