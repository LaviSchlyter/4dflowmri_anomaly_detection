
# Anomaly Detection in 4D Flow MRI's



This project presents a deep learning-based approach for the detection of anomalies in 4D flow MRI data. By employing self-supervised learning and reconstruction-based techniques, our model efficiently identifies abnormal blood flow patterns. The repository contains the code, data processing scripts, and evaluation tools developed during the study.







## Authors

- [@LaviniaSchlyter](https://github.com/LaviSchlyter)


## Run Locally

Clone the project

```bash
  git clone https://github.com/LaviSchlyter/4dflowmri_anomaly_detection
```

Go to the project directory

```bash
  cd 4dflowmri_anomaly_detection
```

Install dependencies

```bash
  conda create --name <env> --file requirements.txt
```

Preprocess the data by running\
Here it is assumed that you have the segmentations already, else you need to run the code from this [repo](https://github.com/LaviSchlyter/4D-Flow-CNN-Segmentation)
```bash
  cd helpers
```

```bash
  python data_bern_numpy_to_preprocessed_hdf5.py
```

Train the network

For the Self-Supervised Learning (SSL) and the Reconstruction-Based (RB) method both use the `config_vae.yaml` to set the parameters in the `config` folder

```bash
  # Run the bash file which launches the `run.py` onto the cluster if available
  sbatch vae_convT_run.sh.sh
```
For the conditionally parametrized version use the `config_cond_vae.yaml` to set the parameters in the `config` folder

```bash
  # Run the bash file which launches the `run.py` onto the cluster if available
  sbatch cond_vae_run.sh.sh.sh
```

Running inference\
To run the best saved model on the test data

```bash
  python visualize_best_model_outputs.py
```

Backtransformation and better visualization
In the `helpers` folder once you have the inference results in the Results directoy run `python backtransform.py`

This will backtransform the anomaly slices into their original reference frame as well as convert them to vtk for easy visualization in 4D using ParaView

