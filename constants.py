### DATA KEYS ###
PATIENT_ID = "patient_id"
HADM_ID = "hadm_id"
STUDY_ID = "study_id"
DICOM_ID = "dicom_id"
XRAY_TIME = "xray_time"
ADMIT_TIME = "admittime"
DISCHARGE_TIME = "discharge_time"
READMISSION_TIME = "readmission_time"

### METADATA FILENAMES ###
OUTCOMES_DATA_FOLDER = "/data/vision/polina/projects/chestxray/bting/outcomes_chestxray/outcomes_data"
CROSS_VALIDATION_FOLDER = OUTCOMES_DATA_FOLDER + "/cross_validation_data"
ALL_METADATA_FILENAME = OUTCOMES_DATA_FOLDER + "/mimic_xrays_outcome.csv"
TRAIN_METADATA_FILENAME = OUTCOMES_DATA_FOLDER + "/mimic_xrays_outcome_train.csv"
TEST_METADATA_FILENAME = OUTCOMES_DATA_FOLDER + "/mimic_xrays_outcome_test.csv"

MIMIC_DATA_FOLDER = '/data/vision/polina/projects/chestxray/bting/outcomes_chestxray/data'
CORE_ADMISSIONS_FILENAME = MIMIC_DATA_FOLDER + '/core/admissions.csv'
CXR_METADATA_FILENAME = MIMIC_DATA_FOLDER + '/mimic-cxr-metadata.csv'

### DATA FOLDERS ###
CHESTXRAY_PNGS_FOLDER = '/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/'

### VIEW POSITION ###
AP = "AP"
PA = "PA"
LATERAL = "LATERAL"