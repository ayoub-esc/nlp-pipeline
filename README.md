# Practicum Project NLP Pipeline for Unstructured Clinical Data 

The model directory cotains a file with the model weights that can be used to load the mode and the encoder file from sklearn.preprocessing.LabelEncoder() to map model out outs to ICD-10 codes.

The data directory contains the file used for validation - test.csv and a sample test file including a small subset of MIMIC-III cinical notes from test.csv that can be uploaded to the pipeline for testing purposes. The test_expected_codes.csv details the expected response. A custom CSV file can be used as long as it is in the same format as test_notes.csv: "clinical note #1" , "clinical note #2"... 

WARNING: Please do not upload large test files as it will crash the deployment.

The XLNet_Model.ipynb was used to create and train the model.

Deployment can be found at: https://clinical-nlp-pipeline.ue.r.appspot.com
