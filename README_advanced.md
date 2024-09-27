# Anomaly Detection in 4D Flow MRI's

This project presents a deep learning-based approach for the detection of anomalies in 4D flow MRI data. By employing self-supervised learning and reconstruction-based techniques, our model efficiently identifies abnormal blood flow patterns. The repository contains the code, data processing scripts, and evaluation tools developed during the study.


Here we present step by step what should be run from final segmentation to cross sectional slices, model results, back-transformation and Paraview visualization.

## Segmentation 
1. Initial manual segmentation: Use GUI segmentation software from Nicolas Blondel 
2. Use CNN segmentation model


### Manual segmentation
1. Go to the folder `/usr/bmicnas02/data-biwi-01/jeremy_students/previous_work/aorta_segmentation/nicolas/4d_flow_mri_segmentation_tool/random walker`
2. In the `functions.py` file, change in the `load_bern_example_data` function the patient which needs to be segmented
3. Go to the _main file_ `random_walker_gui.py` file select "Bern" as dataset
4.  

```bash 
#Run the script, if available, use the `seg` conda environment 
python random_walker_gui.py
```

 

Workflow for best segmentation result:
1) Cycle through display modes ("Overlab Image"-Button) until the image with the velocity magnitude is shown
2) Explore the 4D volume with the sliders "t-axis" and "z-axis" until you see good contrast in the velocity (typically t=3)
3) Press "Scribble FG" to add a scribble to the foregound, draw a couple of lines inside the aorta, you can also contour
4) Press "Scribble BG" for background and repeat
5) T and z axis are frozen (on purpose)
6) Press "Add scribble" to unfreeze and add more scribbles in other slices
7) Add 2-3 more scribbles at other slices ("z-axis"-slider)
8) Press "Run 3D" to run the algorithm in 3D
9) To propagate the 3D result to a 4D segmentation press "Load segm -> markers" (uses 3D segmentation as rough markers for other timesteps with dilation and erosion)
10) Press "Run 4D" to run the algorithm in 4D using the precomputed markers from the 3D segmentation
11) Save the segmentation 
12) Done!

### CNN segmentation
1. Prepare the Freiburg data
```bash
# This is to prepare the Freiburg data
  python data_freiburg_dicom_to_numpy.py
  python data_freiburg_numpy_to_hdf5.py
```
2. Prepare the Bern data
```bash
# This is to prepare the Bern data
  python bern_numpy_to_hdf5_for_training.py
```
3. Go into the `experiments` folder and set the parameters in `unet.py`.

4. Run the bash file
```bash 
sbatch train.sh
```
5. **Inference** We don't have a test set but through inference we run the best saved model on the data from Bern that was not manually segmented. Go into `experiments` and set parameters in `exp_inference.py`.

6. Run inference

```bash 
sbatch bern_inference.sh
```


### Visualize segmentation
In the `visualization` folder you have a notebook that takes the results from the inference network and where you can visualize the results in 3D. There is also a file `segmentation_visualization.py` where given the models the 3D visualization of predictions are saved within the models directory.


## Anomaly detection 
In this part, our main is to detect anomalies within the ascending aorta.

### Pre-processing
We need to pre-process the data by taking the segmented aorta and extracting cross sectional slices. 

1. We first run ````python data_bern_numpy_to_preprocessed.py``` or `sbatch preprocess_bern.sh` in `helpers` directory. By default it will run the preprocessing versoin where we slice and mask the aorta using a dilated version of the segmentation. If needed change, go to main within the file and change the functions called. This will create hdf5 files within the `data` directory. These files will have the shape [#patients*#slices, 32, 32, 24, 4] ([#patients*#slices, x, y, z (along aorta), t, channels]).  The channels contain the components of the velocity as well as the 'intensity'.  The #slices is typically chosen here to be 64. Runing this file creates two additional folders:
a) We save the geometric information of the image through a dictionary for each slice in order to allow for backtransformation (slices are put back in their original image).
b) For each image we create an image with a gradient from foot to head, left to right and back (posterior) to front (anterior). Using the segmetation of the different patients, it will also do the slicing procedure, this is to ensure that what is perceived by anterior by the radiologist will match our anterior as the slicing operation also includes rotation. 

2. The validation process was done through splitting the cross sectional aorta into quadrants. These quadrants are created through running `create_quadrant_masks.py`.

3. As we introduce synthetic anomalies, we need to run `python synthetic_anomalies.py` or `sbatch synthetic_creation.sh` to create and place the anomalies into the data. In the suffix field, place the name of your validation file. **This step finalizes the data setup**.

3. To change the configuration of the run, go to `config/config_cnn.yaml` to set configurations where the following were used for the paper
```yaml
seed: 5 #5,10,15,20,25
suffix_data: '_without_rotation_with_cs_skip_updated_ao_S10_balanced' 
self_supervised: True
use_synthetic_validation: True

# Depending on synthetic anomalies introduced in training we need to remove indices from validation. For the paper, we use in training Poisson blending with gradient mixing in training so we remove in validation. We have 10 subjects in validation.
indices_to_remove: [3200, 3840]
```

4. To train the model:
```python
  # Run the bash file which launches the `run.py` onto the cluster
  cd run_scripts
  sbatch run_SimpleConvNet.sh

  # Logs appear in `sbatch_logs/sbatch_train_logs`
  # Models are saved in `Saved_models` directory
```

5. To run inference

    a) Go to `inference_experiments_list.py`: Create a dictionary named who's name will depend on the data that was used. For eg. `experiments_with_cs`are the experiments that used both sequential and compressed sensing data. See file for other options as this ensures the right indices within h5df file is given

    b) Go to `inference_test.py`: Select the desired parameters under _# Define experimental settings_ on whether to visualize the plots, backtransform or not the results etc... 

    c) Run using:
```python
  # Run
  sbatch inference_test

  # Logs appear in `sbatch_logs/sbatch_inference_logs`
  # Results are stored in `Results/Evaluation/{model}/{preprocess_method}/{experiment}` 
```

The results contain:

a. `log_test.txt`: log of inference run

b. `test` directory with inputs, outputs, backtransformed outputs and all visualizations

c. `subject_df_with_anomaly_scores_{seed}.csv`: file containing ID, Age, sex, anomaly 
scores, p-values of permutation tests, region based level prediction etc

d. `subject_df_with_anomaly_scores_simple.csv`: Simplified version


6. To visualize the aggregated results over seeds: `Results/Aggregated_results_across_seeds/Basic_visualization_across_seeds.ipynb`


# Paraview tutorial - for visualizations
### Visualize the flow and segmentation 

### Smoothen the aorta

### Visualize the backtransformed version of the anomaly scores 

1. Load the backtransformed anomaly scores from the Results/Evaluation/{architecture}/masked_slice (after having it backtransformed in the inference_test.py file)

2. Load the segmentation folder of a subject from "data/inselspital/kady/preprocessed/(controls or patients)"/vtk_scaled/{subject_id} (open seg_...._velocity.vtk)

3. On the segmetation select the _threshold_ filter on magnitude intensity and set the lower threshold to a low value 0.001 for eg. (just to remove the zero) you should be seeing the aorta. You can select a solid color and reduce the opacity

4. You can smoothen the aorta by
- Select 'cells on' to highlight all of the aorrta as a region (you have to select several regions... not everything in one go - use the green plus sign on the layout)
- Use 'extract selection' filter
- Use the 'extract surface' filter
- Use the 'smooth' filter with like 500 of points
- Add a 'ExtractTimeSteps1'for input one, cause smoothing created some holes across time step

5. For the anomaly score, use the _threshold_ filter with above upper threhold of 1e-6 for example, again we want higher than zero. In order to make it smoother, apply the 'Resample to image' filter and view as volume rather than surface






# End
17.09.2024 - Lavinia Schlyter

