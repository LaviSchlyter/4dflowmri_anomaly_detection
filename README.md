# Anomaly Detection in 4D Flow MRI's

This project presents a deep learning-based approach for the detection of anomalies in 4D flow MRI data. By employing self-supervised learning and reconstruction-based techniques, our model efficiently identifies abnormal blood flow patterns. The repository contains the code, data processing scripts, and evaluation tools developed during the study.

## Anomaly detection 
In this part, our main is to detect anomalies within the ascending aorta.

### Pre-processing
We need to pre-process the data by taking the segmented aorta and extracting cross sectional slices. 

1. By running `preprocess.sh `, we run the default preprocessing used for the article where we slice and mask the aorta using a dilated version of the segmetation. You can modify the parameters `config_preprocessing.yaml` in `config` directory.
```bash
cd scripts

sbatch preprocess.sh 

# Logs will appear in `logs/preprocess_logs`
```
This will create hdf5 files within the `data` directory. These files will have the shape [#patients*#slices, 32, 32, 24, 4] ([#patients*#slices, x, y, z (along aorta), t, channels]).  The channels contain the components of the velocity as well as the 'intensity'.  The #slices is typically chosen here to be 64.
  
a) We save the geometric information of the image through a dictionary for each slice in order to allow for backtransformation (slices are put back in their original image). `geometry_for_backtransformation`
b) For each image we create an image with a gradient from foot to head, left to right and back (posterior) to front (anterior). Using the segmetation of the different patients, it will also do the slicing procedure, this is to ensure that what is perceived by anterior by the radiologist will match our anterior as the slicing operation also includes rotation. `gradient_matching`
c) The regional validation process was done through comparing quadrants to Inselspital. Quadrants for each patient is stored in `quadrants_between_axes` and `quadrants_main_axes`

### Training model

2. To change the configuration of the run (epochs, seed, self-supervised, etc..), go to `config/config_cnn.yaml`, to change the preprocess method or model to run, go into `scripts/train.sh` (only CNN model is given for article).

3. To train the model:
```python
  cd scripts
  sbatch train.sh

# Logs appear in `logs/train_logs`
# Models are saved in `Saved_models` directory
```

### Running inference
 
 4. Configure the inference process via the `config/config_inference.yaml` file. Key parameters include model selection, preprocessing details, and visualization controls, among others. Here’s a brief overview of essential settings:

- **model_name**: Specifies the model to be used for inference.
- **preprocess_method**: Defines the method for data preprocessing.
- **visualize_inference_plots**: Toggle to enable/disable the generation of plot outputs during inference.

5. Run the script with 
```bash
cd scripts

sbatch inference.sh

# Logs appear in `logs/inference_logs`
# Results are shown in `Results` directory
```

The results contain:

a. `log_test.txt`: log of inference run

b. `test` directory with inputs, outputs, backtransformed outputs and all visualizations

c. `subject_df_with_anomaly_scores_{seed}.csv`: file containing ID, Age, sex, anomaly 
scores, p-values of permutation tests, region based level prediction etc

d. `subject_df_with_anomaly_scores_simple.csv`: Simplified version

6. To visualize the aggregated results over seeds: `Results/Aggregated_results_across_seeds/Basic_visualization_across_seeds.ipynb`


 




