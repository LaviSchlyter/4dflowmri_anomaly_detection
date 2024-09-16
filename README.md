
# Anomaly Detection in 4D Flow MRI's



This project presents a deep learning-based approach for the detection of anomalies in 4D flow MRI data. By employing self-supervised learning and reconstruction-based techniques, our model efficiently identifies abnormal blood flow patterns. The repository contains the code, data processing scripts, and evaluation tools developed during the study.

## Authors

- [@LaviniaSchlyter](https://github.com/LaviSchlyter)


## How to run - simple version

#### Config run

Got to `config/config_cnn.yaml` to set configurations. 
```yaml
seed: 5 #5,10,15,20,25
suffix_data: '_without_rotation_with_cs_skip_updated_ao_S10_balanced' # used for paper
# If SSL
self_supervised: True
use_synthetic_validation: True
# If RB
self_supervised: False
use_synthetic_validation: False

# Depending on synthetic anomalies introduced in training we need to remove indices from validation. For the paper, we use in training Poisson blending with gradient mixing in training so we remove in validation. We have 10 subjects in validation.
# Go in utils.py `apply_blending` function, if desire to change 
indices_to_remove: [3200, 3840]

```

#### Run bash script
```python
  # Run the bash file which launches the `run.py` onto the cluster
  cd run_scripts
  sbatch run_SimpleConvNet.sh

  # Logs appear in `sbatch_logs/sbatch_train_logs`
```
#### Running inference
Models are saved in `Saved_models` directory

- Go to `inference_experiments_list.py`: Create a dictionary with the names of the experiments under the name ```short_experiments_with_cs``` or without the `short` (this will give the correct indices on the dataset used). `with_cs` extension because we also include compressed sensing data

- Go to `inference_test.py`. Select the desired parameters. `backtransform_anomaly_scores_bool`, `visualize_inference_plots` etc

```python
  # Run
  sbatch inference_test

  # Logs appear in `sbatch_logs/sbatch_inference_logs`
```

#### View results

Results are stored in `Results/Evaluation/{model}/{preprocess_method}/{experiment}` 

**for eg.** `Results/Evaluation/simple_conv/masked_slice/20240530-1325_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3`

_They contain:_

1. `log_test.txt`: log of inference run
2. `test` folder with inputs, outputs, backtransformed outputs and all visualizations
3. `subject_df_with_anomaly_scores_{seed}.csv`: file containing ID, Age, sex, anomaly scores, p-values of permutation tests, region based level prediction etc
4. `subject_df_with_anomaly_scores_simple.csv`: Simplified version


#### View aggregated results over seeds
Notebook to view results simple results across seeds
`Results/Aggregated_results_across_seeds/Basic_visualization_across_seeds.ipynb`


## Anomaly detection pipeline
Here we present step by step what should be run from final segmentation to cross sectional slices, model results, back-transformation and Paraview visualization.



