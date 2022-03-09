import argparse
import logging
import os

from datetime import datetime as dt
import pandas as pd


current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()

parser.add_argument('--core_dir', type=str,
					default='/data/vision/polina/projects/chestxray/bting/outcomes_chestxray/data/core/admissions.csv',
					help='The image data directory')
parser.add_argument('--cxr_metadata', type=str,
					default='/data/vision/polina/projects/chestxray/bting/outcomes_chestxray/data/mimic-cxr-metadata.csv',
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
                try:
                    xray_datetime = dt.strptime(str(xray['StudyDate']) + " " + str(xray['StudyTime']).split(".")[0], DATETIME_FORMAT_FOR_XRAYS)
                except ValueError:
                    # handle case where datetime is formatted as %Y%m%d %S"
                    xray_datetime = dt.strptime(str(xray['StudyDate']) + " 000" + str(xray['StudyTime']).split(".")[0], DATETIME_FORMAT_FOR_XRAYS)
                if admit_time <= xray_datetime <= discharge_time:
                    discharge_dicom_id = xray['dicom_id']
                    hadm_id = admission_info.iloc[i]['hadm_id']
                    readmission_data.append([id, hadm_id, discharge_dicom_id, xray_datetime, admit_time, readmission_time])

    readmission_df = pd.DataFrame(readmission_data, columns = ['patient_id', 'hadm_id', 'dicom_id', 'xray_time', 'admittime', 'readmission_time'])
    readmission_df.to_csv("./mimic_xrays_outcome.csv", index=False)

    return readmission_df

preprocess_data()
