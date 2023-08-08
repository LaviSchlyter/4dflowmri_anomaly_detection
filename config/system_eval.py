import os


# ==================================================================
# SET THESE PATHS MANUALLY
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['biwidl203']

# ==================================================================
# project dirs
# ==================================================================
project_code_root = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/4dflowmri_anomaly_detection/'
project_data_root = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady'

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'logs/')
