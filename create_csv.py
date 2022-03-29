'''
Author: Britney Ting

Script creating .csv file
that links CXR-MIMIC xray and 
CXR-IV readmission data.
'''

import argparse
import logging
import os

from datetime import datetime as dt
from tabnanny import check
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from constants import *


current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()

parser.add_argument('--core_dir', type=str,
					default=CORE_ADMISSIONS_FILENAME,
					help='The image data directory')
parser.add_argument('--cxr_metadata', type=str,
					default=CXR_METADATA_FILENAME,
					help='The image data directory')

def preprocess_data():
    '''
    Construct .csv file that contains patient chest xray corresponding to readmission time,
    where readmission time is the difference between the discharge time and next admission time. 
    The filename is "mimic_xrays_outcome.csv".
    '''
    DATETIME_FORMAT_WITH_SECONDS = "%Y-%m-%d %H:%M:%S"
    DATETIME_FORMAT_FOR_XRAYS = "%Y%m%d %H%M%S"

    args = parser.parse_args()
    core_df = pd.read_csv(args.core_dir)
    cxr_metadata_df = pd.read_csv(args.cxr_metadata)
    unique_patients = set(core_df['subject_id'])

    readmission_data = []
    for id in unique_patients:
        # sort patient's info by dischtime
        admission_info = core_df.loc[core_df['subject_id'] == id].sort_values(by='dischtime')
        num_admissions = admission_info.shape[0]
        xrays = cxr_metadata_df.loc[cxr_metadata_df['subject_id'] == id]
        for i in range(num_admissions-1):
            admit_time = dt.strptime(admission_info.iloc[i]['admittime'], DATETIME_FORMAT_WITH_SECONDS)
            discharge_time = dt.strptime(admission_info.iloc[i]['dischtime'], DATETIME_FORMAT_WITH_SECONDS)
            next_admit_time = dt.strptime(admission_info.iloc[i+1]['admittime'], DATETIME_FORMAT_WITH_SECONDS)
            readmission_time = next_admit_time - discharge_time
            for index, xray in xrays.iterrows():
                discharge_dicom_id = xray[DICOM_ID]
                study_id = xray[STUDY_ID]
                # Only use anteroposterior or posteroanterior views
                if xray['ViewPosition'] != PA and xray['ViewPosition'] != AP:
                    continue
                # Can't use image that doesn't exist in Ray's scratch folder
                elif not os.path.isfile(os.path.join(CHESTXRAY_PNGS_FOLDER, f"p{id}_s{study_id}_{discharge_dicom_id}.png")):
                    continue
                try:
                    xray_datetime = dt.strptime(str(xray['StudyDate']) + " " + str(xray['StudyTime']).split(".")[0], DATETIME_FORMAT_FOR_XRAYS)
                except ValueError:
                    # handle case where datetime is formatted as %Y%m%d %S"
                    xray_datetime = dt.strptime(str(xray['StudyDate']) + " 000" + str(xray['StudyTime']).split(".")[0], DATETIME_FORMAT_FOR_XRAYS)
                if admit_time <= xray_datetime <= discharge_time:
                    hadm_id = admission_info.iloc[i][HADM_ID]
                    readmission_data.append([id, hadm_id, study_id, discharge_dicom_id, xray_datetime, admit_time, readmission_time])

    readmission_df = pd.DataFrame(readmission_data, columns = [PATIENT_ID, HADM_ID, STUDY_ID, DICOM_ID, XRAY_TIME, ADMIT_TIME, READMISSION_TIME])
    readmission_df.to_csv(ALL_METADATA_FILENAME, index=False)

    return readmission_df

# preprocess_data() # uncomment to generate full .csv

def check_no_crossing_ids(train_file: str, test_file: str):
    '''
    Checks that patients from training set aren't in test set
    '''
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    patients_train = set(train_df[PATIENT_ID])
    patients_test = set(test_df[PATIENT_ID])
    return patients_train.intersection(patients_test) == set()

# print(check_no_crossing_ids(TRAIN_METADATA_FILENAME, TEST_METADATA_FILENAME))

def split_train_test(filename: str):
    '''
    Takes in the preprocessed .csv file (generated using preprocess_data())
    and splits it train and test .csv files.
    90% train and 10% split along unique patient ids.
    '''
    df = pd.read_csv(filename)
    splitter = GroupShuffleSplit(test_size=.10, random_state=42)
    split = splitter.split(df, groups=df[PATIENT_ID])
    train_inds, test_inds = next(split)
    train_df = df.iloc[train_inds]
    test_df = df.iloc[test_inds]

    train_df.to_csv(TRAIN_METADATA_FILENAME, index=False)
    test_df.to_csv(TEST_METADATA_FILENAME, index=False)

    assert(check_no_crossing_ids(TRAIN_METADATA_FILENAME, TEST_METADATA_FILENAME))

# split_train_test(ALL_METADATA_FILENAME)

def check_all_images_available(files: list):
    for file in files:
        unavailable = 0
        df = pd.read_csv(file)
        for index, xray in df.iterrows():
            patient_id = xray[PATIENT_ID]
            study_id = xray[STUDY_ID]
            img_id = xray[DICOM_ID]
            png_path = os.path.join(CHESTXRAY_PNGS_FOLDER, f"p{patient_id}_s{study_id}_{img_id}.png")
            if not os.path.isfile(png_path):
                unavailable += 1
        print(f"No image found for {unavailable}/{df.shape[0]} dicom_ids for {file}.")

# check_all_images_available([TRAIN_METADATA_FILENAME, TEST_METADATA_FILENAME])

