'''
Author: Britney Ting

Instantiates classes needed for building dataloaders.
Borrowed from Ruizhi Liao with permission.
'''

import cv2
import glob
from datetime import datetime as dt, timedelta
import numpy as np
import os
import pandas as pd
import torchvision

from constants import *

class MimicID:
	subject_id = ''
	study_id = ''
	dicom_id = ''

	def __init__(self, subject_id, study_id, dicom_id):
		self.subject_id = str(subject_id)
		self.study_id = str(study_id)
		self.dicom_id = str(dicom_id)

	def __str__(self):
		return f"p{self.subject_id}_s{self.study_id}_{self.dicom_id}"

class CXRImageDataset(torchvision.datasets.VisionDataset):
    """A CXR iamge dataset class that loads png images 
    given a metadata file and return images and labels 

    Args:
        data_dir (string): Root directory for the CXR images.
        dataset_metadata (string): File path of the metadata 
            that will be used to contstruct this dataset. 
            This metadata file should contain data IDs that are used to
            load images and labels associated with data IDs.
        data_key (string): The name of the column that has image IDs.
        label_key (string): The name of the column that has labels.
        transform (callable, optional): A function/tranform that takes in an image 
            and returns a transfprmed version.
    """
    
    def __init__(self, data_dir, dataset_metadata, 
                 data_key=DICOM_ID, label_key=READMISSION_TIME,
    			 transform=None, cache=False):
        super(CXRImageDataset, self).__init__(root=None, transform=transform)
        self.data_dir = data_dir
        self.dataset_metadata = pd.read_csv(dataset_metadata)
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.image_ids = self.dataset_metadata[data_key]
        self.select_valid_labels()
        self.cache = cache
        if self.cache:
            self.cache_dataset() 
        else:
            self.images = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        patient_id, study_id, img_id, label = self.dataset_metadata.loc[idx, [PATIENT_ID, STUDY_ID, self.data_key, self.label_key]]

        days, _, time = label.split(" ")
        hours, minutes, seconds = time.split(":")
        label = int(int(days) < 8)

        if self.cache:
            img = self.images[str(idx)]
        else:
            png_path = os.path.join(self.data_dir, f"p{patient_id}_s{study_id}_{img_id}.png")
            # print("png_path:", png_path)
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)

        if self.transform is not None:
            img = self.transform(img)

        img = np.expand_dims(img, axis=0)

        return img, label, img_id

    def select_valid_labels(self):
        self.dataset_metadata[XRAY_TIME] = self.dataset_metadata[XRAY_TIME].apply(lambda time: dt.strptime(time, "%Y-%m-%d %H:%M:%S"))
        self.dataset_metadata[DISCHARGE_TIME] = self.dataset_metadata[DISCHARGE_TIME].apply(lambda time: dt.strptime(time, "%Y-%m-%d %H:%M:%S"))
        self.dataset_metadata = self.dataset_metadata.loc[self.dataset_metadata.groupby([PATIENT_ID, ADMIT_TIME])[XRAY_TIME].idxmax()] # select for patients' last image only
        # self.dataset_metadata = self.dataset_metadata.loc[self.dataset_metadata.groupby([PATIENT_ID, ADMIT_TIME])[XRAY_TIME].idxmin()] # select for patients' first image only
        # self.dataset_metadata = self.dataset_metadata.loc[self.dataset_metadata.groupby([PATIENT_ID, ADMIT_TIME])[XRAY_TIME].apply(lambda x: x.sample(1))] # TODO: [MUST CHECK THAT THIS IS CORRECT] select random image from patient's visit
        # XRAY_TIMEDELTA_BEFORE_DISCHARGE = self.dataset_metadata[DISCHARGE_TIME] - self.dataset_metadata[XRAY_TIME]
        self.dataset_metadata = self.dataset_metadata.loc[self.dataset_metadata[XRAY_TIME] >= self.dataset_metadata[DISCHARGE_TIME] - pd.Timedelta(days=2)] # select for xrays that were taken at most 48 hours prior to discharge

        self.dataset_metadata = self.dataset_metadata.reset_index(drop=True)
        self.image_ids = self.dataset_metadata[self.data_key]

    def cache_dataset(self):
        for idx in range(self.__len__()):
            img_id, label = self.dataset_metadata.loc[idx, [self.data_key, self.label_key]]
            png_path = os.path.join(self.data_dir, f'{img_id}.png')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
            if idx == 0:
                self.images = {}
            self.images[str(idx)] = img

    ##########################################
    ### TODO: MUST MODIFY TO SUIT MY MODEL ###
    ##########################################
    @staticmethod
    def create_dataset_metadata(mimiccxr_metadata, label_metadata, save_path,
                                data_key='study_id', label_key=['edema_severity'],
                                mimiccxr_selection={'view': ['frontal']},
                                holdout_metadata=None, holdout_key='subject_id'):
        """Create a dataset metadata file for CXRImageDataset 
        given a MIMIC-CXR metadata file and a label metadata file.
        """

        mimiccxr_metadata = pd.read_csv(mimiccxr_metadata)
        label_metadata = pd.read_csv(label_metadata)

        dataset_metadata = mimiccxr_metadata[mimiccxr_metadata[data_key].isin(label_metadata[data_key])]

        if mimiccxr_selection != None:
            for key in mimiccxr_selection:
                dataset_metadata = dataset_metadata[dataset_metadata[key].isin(mimiccxr_selection[key])]

        if holdout_metadata != None:
            holdout_metadata = pd.read_csv(holdout_metadata)
            dataset_metadata = dataset_metadata[~dataset_metadata[holdout_key].isin(holdout_metadata[holdout_key])]

        label_metadata = label_metadata[[data_key]+label_key]
        dataset_metadata = dataset_metadata.merge(label_metadata, left_on=data_key, right_on=data_key)

        dataset_metadata['mimic_id'] = dataset_metadata.apply(lambda row: \
            MimicID(row['subject_id'], row['study_id'], row['dicom_id']).__str__(), axis=1)
        dataset_metadata = dataset_metadata[['mimic_id']+label_key]

        dataset_metadata.to_csv(save_path, index=False)
