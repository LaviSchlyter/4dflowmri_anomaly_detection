import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

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
#orig_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/source_data'

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'logs/')
