import os
import numpy as np
import SimpleITK as sitk

output_path = "/Volumes/G-DRIVE mobile/KaggleData/output"

baseRoot, dirs, files = next(os.walk('/Users/admin/Documents/CS676/Lungs/sample/testSlices'))

for patient in dirs:

    patient_path = baseRoot + "/" + patient

    patientID = patient[:-6]
    num_slices = len(os.listdir(patient_path)) - 1

    full_img = np.ndarray([num_slices, 480, 640, 4], dtype=np.float32)
    i = 0

    for file in os.listdir(patient_path):
        print file
        img = sitk.ReadImage(patient_path + "/" + file)
        full_img[i] = sitk.GetArrayFromImage(img)
        i =+ 1
    np.save(os.path.join(output_path, "images_%32s.npy" % patientID), full_img)
#
# # Get the list of directories (patients)
# for baseRoot, dirs, files in os.walk('/Users/admin/Documents/CS676/Lungs/sample/slices'):
#
#     print "baseRoot: " + baseRoot
#
#     patientID = baseRoot[49:] # remove the filepath
#     patientID = patientID[:-6] # remove the label
#     print patientID
#
#     num_slices = len(files) - 1 # subtract one so that it doesn't include the directory itself
#
#     full_img = np.ndarray([num_slices, 512, 512], dtype=np.float32)
#     i = 0
#
#     for f in files:
#         print "     " + f
#         if str(f)[0] != '.':
#             img = sitk.ReadImage(f)
#             full_img[i] = sitk.GetArrayFromImage(img)
#             i =+ 1
#     np.save(os.path.join(output_path, "images_%32s.npy" % patientID), full_img)
