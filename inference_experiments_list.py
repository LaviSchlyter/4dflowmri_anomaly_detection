
############################################## CORRECT VALIDATION SET ##############################################

## 1. WITHOUT UPDATED ASCENDING AORTA ! We should really do it becsaue the centerlines don't look great ----------------------------------------------------------------------- BLUE TABLE

## The validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.
# -----------> HERE we have corrected the indices we remove on the validation set

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231217-1536_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231217-1532_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231215-1433_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231217-1542_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231217-1538_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231216-0857_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231218-1017_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_decreased_interpolation_factor_cube_3']
                                        
"""

## 2. WITH UPDATED ASCENDING AORTA, WITHOUT ALL SUBJECTS (SET TO ZERO) -------------------------------------------------------------------------------------------------------- PURPLE TABLE

# We don't have all the subjects because angle 3/2 did not work for some subjects. Those two subjects are zero basically.
# We ONLY remove the first 2 slices (and use skip)
## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.
# -----------> HERE we have corrected the indices we remove on the validation set

# Little note: In the first case, I hasn't yet renamed the dataset that's why there is no zero in the name.
"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231215-1216_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231217-1605_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version_with_zeros_with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

# True one
#name_pre_extension = ['_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10',
#                      '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10']

# For testing on another test case
#name_pre_extension = ['_with_rotation_without_cs', '_with_rotation_without_cs', '_without_rotation_without_cs', '_without_rotation_without_cs']

name_pre_extension = ['_with_rotation_without_cs_skip_updated_ao_S10', '_with_rotation_without_cs_skip_updated_ao_S10', '_without_rotation_without_cs_skip_updated_ao_S10', '_without_rotation_without_cs_skip_updated_ao_S10']

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231218-1010_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version_with_zeros_without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231217-1609_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version_with_zeros_without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""


## 3. WITH UPDATED ASCENDING AORTA, WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------------------------------------ YELLOW TABLE

## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order
# -----------> HERE we have corrected the indices we remove on the validation set

"""
list_of_experiments_with_rotation = [
                                    'vae_convT/masked_slice/20231215-1034_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231214-1948_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231213-2247_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231214-1945_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
name_pre_extension = ['']# This is the true thing


name_pre_extension = ['_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10','_without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10', 
                      '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10', '_without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10']

list_of_experiments_without_rotation = [
                                        'vae_convT/masked_slice/20231215-1032_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231214-1951_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231213-2252_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231214-1923_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',]

"""


## 4. WITH UPDATED ASCENDING AORTA, WITHOUT ALL SUBJECTS (WE REMOVED THEM) ------------------------------------------------------------------------------------------------------ RED TABLE

## Also the validation indices are not correct. We removed only without mixing from validation but used the mixed ones as well for training.

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231217-1709_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231213-2224_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20231213-2230_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     ]
#name_pre_extension = ['_with_rotation_without_cs_skip_updated_ao_S10',
#                      '_with_rotation_without_cs_skip_updated_ao_S10',
#                      '_without_rotation_without_cs_skip_updated_ao_S10',
#                      '_without_rotation_without_cs_skip_updated_ao_S10'] # When you want to test on other test case.
name_pre_extension = ['']
# No need to add extension, it is present in the name of the experiment.    

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231217-1702_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231213-2227_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231213-2229_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_some_data_redid_other_version__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""
"""

## 5. WITH UPDATED ASCENDING AORTA, WITH POISSON MIX FOR TRAINING,  WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------ GRAY TABLE
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order
# -----------> HERE we have corrected the indices we remove on the validation set
# POisson mix for training 

list_of_experiments_with_rotation = [
                                    'vae_convT/masked_slice/20231220-2300_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231220-2256_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231215-1144_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231215-1109_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231219-1926_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231219-1930_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
                                    
name_pre_extension = [''] # This is the true thing

list_of_experiments_without_rotation = [
                                        'vae_convT/masked_slice/20231220-2302_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2305_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231215-1150_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231215-1139_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231218-1031_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        ]

"""
## 6. WITH UPDATED ASCENDING AORTA, WITH AND WITHOUT GRADIENT MIX POISSON FOR TRAINING BUT ALSO VALIDATION,  WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------- PINK TABLE
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order
# -----------> HERE we have either ways no indices
# Training with poisson mix adn wihtout mix and validation too - no removing indices


"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20231221-1840_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1811_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1757_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1752_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                    'vae_convT/masked_slice/20231221-1750_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
                                    
name_pre_extension = [''] # This is the true thing

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20231220-2317_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2321_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2325_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231220-2329_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20231228-1203_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
                                        ]
"""



## 7. RECONSTRUCTION BASEDWITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------ GRAY TABLE
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order


"""

list_of_experiments_with_rotation = [
                                        'vae_convT/masked_slice/20240108-1859_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_with_rotation',
                                        'vae_convT/masked_slice/20240108-1853_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_with_rotation',
                                        'vae_convT/masked_slice/20240108-1857_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_with_rotation',
                                        'vae_convT/masked_slice/20240108-1600_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_with_rotation',
                                        'vae_convT/masked_slice/20240108-1558_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_with_rotation',
                                        ]



name_pre_extension = ['_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10','_with_rotation_without_cs_skip_updated_ao_S10',
                    '_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10','_without_rotation_without_cs_skip_updated_ao_S10'] # This is the true thing

list_of_experiments_without_rotation = [
                                    'vae_convT/masked_slice/20240108-1118_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_without_rotation',
                                    'vae_convT/masked_slice/20240108-1120_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_without_rotation',
                                    'vae_convT/masked_slice/20240108-1123_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_without_rotation',
                                    'vae_convT/masked_slice/20240108-1521_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_without_rotation',
                                    'vae_convT/masked_slice/20240108-1523_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_without_rotation'
                                    ]
                                    
"""
idx_start_ts = 0
idx_end_ts = 34


############################################## ONLY COMPRESSED SENSING ##############################################
## 7. ONLY COMPRESSED SENSING - WITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------ 
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only 
## We DO NOT order the points in ascending order - ONLY COMPRESSED SENSING

# THE SEEDS HAVE BEEN FIXED


"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240214-1610_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1613_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1615_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1617_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240214-1622_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240215-0858_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0901_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0906_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0910_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240215-0914_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""

"""

## TRYING WITH A DEEPER AUXIALIARY NETWORK
list_of_experiments_with_rotation = ['deep_conv_with_aux/masked_slice/20240307-1319_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240307-1321_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240307-1324_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240307-1459_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240307-1506_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['deep_conv_with_aux/masked_slice/20240308-1033_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240308-1036_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240308-1037_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240308-1040_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240308-1042_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']



"""
"""
## TRYING WITH A DEEPER AUXIALIARY NETWORK
list_of_experiments_with_rotation = ['deep_conv_enc_dec_aux/masked_slice/20240307-1337_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240307-1402_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240307-1421_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240307-1457_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240307-1504_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['deep_conv_enc_dec_aux/masked_slice/20240308-1043_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240308-1044_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240308-1046_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240308-1048_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240308-1106_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""

"""


## TRYING WITH A larger bottle neck with deeper AUXIALIARY NETWORK
list_of_experiments_with_rotation = ['deeper_conv_enc_dec_aux/masked_slice/20240313-1452_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_conv_enc_dec_aux/masked_slice/20240313-1454_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_conv_enc_dec_aux/masked_slice/20240314-1141_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_conv_enc_dec_aux/masked_slice/20240315-0825_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_conv_enc_dec_aux/masked_slice/20240315-0827_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['deeper_conv_enc_dec_aux/masked_slice/20240313-1456_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_conv_enc_dec_aux/masked_slice/20240313-1502_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_conv_enc_dec_aux/masked_slice/20240314-1141_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_conv_enc_dec_aux/masked_slice/20240315-0829_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_conv_enc_dec_aux/masked_slice/20240315-0831_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""


## TRYING WITH A larger bottle neck with deeper AUXIALIARY NETWORK and batch normalization
"""

list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240313-1504_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1131_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1133_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0834_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0837_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240313-1506_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1137_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1139_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0835_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0847_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""


"""

# Same as below but without unwrapping of phases (should be very similar to original)
list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240415-1746_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_norm_post_seg_mag_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240415-1748_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_norm_post_seg_mag_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = []
"""

"""
#Quick trial using unwrapping of phases to deal with aliasing
list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240415-1750_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_norm_post_seg_mag_unwrapped_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240415-1752_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_norm_post_seg_mag_unwrapped_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = []
"""

### CENTERED NORMALIZATION - corrected for gaussian axis (no blur over the 3 axis)


#Quick trial using unwrapping of phases to deal with aliasing
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240417-1541_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240418-0946_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240423-1716_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240423-1719_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240423-1750_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = []



"""
# Same as below but without unwrapping of phases (should be very similar to original)
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240417-1559_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240418-1008_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240423-1329_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240423-1722_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240423-1724_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = []

"""

"""

#Quick trial using unwrapping of phases to deal with aliasing
list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1458_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1502_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1538_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1320_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1322_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1800_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1803_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1804_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1805_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1807_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3']

"""

"""
# Same as below but without unwrapping of phases (should be very similar to original)
list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1449_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1500_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1504_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1106_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1324_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3'
                                     ]

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1755_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
                                        ]
"""


### END OF CENTERED NORMALIZATION - corrected for gaussian axis (no blur over the 3 axis)
"""
#Quick trial using the origianl data to see
list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240416-1148_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240416-1150_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = []
"""


# WE TRAIN WITH BOTH POISSON MIX AND WITHOUT POISSON MIX - we remove both from validation
"""
# WE TRAIN WITH BOTH POISSON MIX AND WITHOUT POISSON MIX - we remove both from validation
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240305-1157_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1158_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1200_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1201_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1422_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240306-0950_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0952_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0953_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0957_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-1000_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""


# WE TRAIN WITH BOTH POISSON MIX AND WITHOUT POISSON MIX - we remove both from validation
"""
# WE TRAIN WITH BOTH POISSON MIX AND WITHOUT POISSON MIX - we remove none from validation 
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240305-1204_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1205_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1209_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1211_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240305-1426_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 17

list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240306-0927_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0930_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0943_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0947_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240306-0948_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""

"""
list_of_experiments_with_rotation =    ['conv_with_aux/masked_slice/20240214-1119_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1120_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1123_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1125_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1128_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


idx_start_ts = 0
idx_end_ts = 17

name_pre_extension = ['']


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240214-1214_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1216_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1217_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1219_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240214-1433_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']



"""
"""

list_of_experiments_with_rotation =    ['conv_enc_dec_aux/masked_slice/20240215-0930_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240215-0957_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1218_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1221_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1223_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


idx_start_ts = 0
idx_end_ts = 17

name_pre_extension = ['']


list_of_experiments_without_rotation = ['conv_enc_dec_aux/masked_slice/20240218-1225_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240218-1226_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1428_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1429_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1433_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""


#############

############################################## ONLY NON COMPRESSED SENSING ##############################################

# THE SEEDS HAVE BEEN FIXED - ONLY NON COMPRESSED SENSING DATA

"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240228-1330_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240228-1333_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240228-1335_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240228-1337_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240228-1342_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 34


list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240228-1318_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240228-1320_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240228-1322_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240228-1324_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240228-1328_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""


"""
list_of_experiments_with_rotation = ['conv_with_aux/masked_slice/20240229-1027_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240229-1029_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240229-1039_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240229-1041_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240229-1043_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 34


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240229-1046_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240229-1049_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240229-1051_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240229-1053_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240229-1107_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""


"""
list_of_experiments_with_rotation = ['conv_enc_dec_aux/masked_slice/20240304-1656_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240304-1701_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240304-1703_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240304-1706_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240304-1708_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 34


list_of_experiments_without_rotation = ['conv_enc_dec_aux/masked_slice/20240304-1710_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240304-1713_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240304-1716_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240304-1723_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240304-1727_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""


"""
list_of_experiments_with_rotation = ['deep_conv_enc_dec_aux/masked_slice/20240312-1208_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240312-1219_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240312-1224_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240319-1324_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240319-1328_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 34


list_of_experiments_without_rotation = ['deep_conv_enc_dec_aux/masked_slice/20240312-1221_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240312-1224_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240319-1331_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240319-1339_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240319-1343_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""
"""

list_of_experiments_with_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240319-1411_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240319-1418_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-1215_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-1219_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-1553_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 34


list_of_experiments_without_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240321-1254_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-1303_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-1355_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-2347_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-2352_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']


"""


######################################### ALL DATA (includes compressed sensing) ##########################################
## 8. ALL DATA - WITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only
## We DO NOT order the points in ascending order


# THE SEEDS HAVE BEEN FIXED - ALL DATA
"""
list_of_experiments_with_rotation = ['vae_convT/masked_slice/20240218-1233_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1236_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1237_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1240_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'vae_convT/masked_slice/20240218-1242_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['vae_convT/masked_slice/20240225-0800_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0803_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0808_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0811_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'vae_convT/masked_slice/20240225-0828_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""


"""
list_of_experiments_with_rotation = ['conv_with_aux/masked_slice/20240225-0833_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-0838_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-2241_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-2243_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_with_aux/masked_slice/20240225-2300_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['conv_with_aux/masked_slice/20240225-2310_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2316_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2318_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2320_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_with_aux/masked_slice/20240225-2322_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""
"""
list_of_experiments_with_rotation = ['conv_enc_dec_aux/masked_slice/20240226-1739_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1741_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1746_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1749_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'conv_enc_dec_aux/masked_slice/20240226-1834_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['conv_enc_dec_aux/masked_slice/20240226-1837_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240226-1845_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240226-1846_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1414_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'conv_enc_dec_aux/masked_slice/20240227-1417_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""

"""
list_of_experiments_with_rotation =   ['deep_conv_with_aux/masked_slice/20240311-1626_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240311-1632_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240311-1637_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240311-1646_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_with_aux/masked_slice/20240311-1648_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['deep_conv_with_aux/masked_slice/20240311-1650_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240311-1653_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240311-1655_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240311-1658_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_with_aux/masked_slice/20240311-1700_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""

"""

list_of_experiments_with_rotation =   ['deep_conv_enc_dec_aux/masked_slice/20240312-1134_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240312-1136_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240319-1350_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240319-1357_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deep_conv_enc_dec_aux/masked_slice/20240319-1401_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['deep_conv_enc_dec_aux/masked_slice/20240312-1139_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240312-1141_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240323-1628_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240323-1632_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deep_conv_enc_dec_aux/masked_slice/20240323-1635_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

"""


"""

list_of_experiments_with_rotation =   ['deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1643_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0822_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0825_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0828_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                     'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0830_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

name_pre_extension = ['']# This is the true thing
idx_start_ts = 0
idx_end_ts = 54


list_of_experiments_without_rotation = ['deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1641_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1638_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-0856_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-0902_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
                                        'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-0905_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']
"""




experiments_only_cs = {
    "with_rotation": [[
        'vae_convT/masked_slice/20240214-1610_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240214-1613_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240214-1615_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240214-1617_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240214-1622_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['deep_conv_with_aux/masked_slice/20240307-1319_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240307-1321_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240307-1324_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240307-1459_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240307-1506_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['deep_conv_enc_dec_aux/masked_slice/20240307-1337_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240307-1402_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240307-1421_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240307-1457_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240307-1504_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['deeper_conv_enc_dec_aux/masked_slice/20240313-1452_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240313-1454_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240314-1141_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240315-0825_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240315-0827_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240313-1504_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1131_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1133_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0834_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0837_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],

    ['vae_convT/masked_slice/20240417-1541_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240418-0946_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240423-1716_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240423-1719_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240423-1750_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3'
        ],
    ['vae_convT/masked_slice/20240417-1559_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240418-1008_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240423-1329_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240423-1722_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240423-1724_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3'
        ],
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1458_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1502_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1538_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1320_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1322_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3'
        ],
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1449_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1500_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240417-1504_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1106_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1324_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3'
        ],
    ['vae_convT/masked_slice/20240305-1157_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1158_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1200_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1201_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1422_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],

    ['vae_convT/masked_slice/20240305-1204_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1205_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1209_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1211_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240305-1426_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['conv_with_aux/masked_slice/20240214-1119_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1120_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1123_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1125_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1128_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['conv_enc_dec_aux/masked_slice/20240215-0930_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240215-0957_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240218-1218_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240218-1221_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240218-1223_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ]

    ],
    "without_rotation": [[
        'vae_convT/masked_slice/20240215-0858_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0901_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0906_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0910_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0914_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['deep_conv_with_aux/masked_slice/20240308-1033_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240308-1036_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240308-1037_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240308-1040_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_with_aux/masked_slice/20240308-1042_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['deep_conv_enc_dec_aux/masked_slice/20240308-1043_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240308-1044_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240308-1046_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240308-1048_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240308-1106_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['deeper_conv_enc_dec_aux/masked_slice/20240313-1456_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240313-1502_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240314-1141_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240315-0829_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_conv_enc_dec_aux/masked_slice/20240315-0831_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240313-1506_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1137_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240314-1139_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0835_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240315-0847_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    [],
    [],
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1800_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1803_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1804_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1805_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1807_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3'
        ],
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240423-1755_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
        ],

    ['vae_convT/masked_slice/20240306-0950_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0952_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0953_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0957_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-1000_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['vae_convT/masked_slice/20240306-0927_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0930_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0943_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0947_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240306-0948_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_all_poisson_tr_val_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['conv_with_aux/masked_slice/20240214-1214_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1216_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1217_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1219_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240214-1433_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
    ['conv_enc_dec_aux/masked_slice/20240218-1225_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240218-1226_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240227-1428_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240227-1429_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240227-1433_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ]

]
}



# WE TRAIN WITH BOTH POISSON MIX AND WITHOUT POISSON MIX - we remove both from validation (all_poisson)

# WE TRAIN WITH BOTH POISSON MIX AND WITHOUT POISSON MIX - we remove both from validation (_all_poisson_tr_val)




############################################## ONLY NON COMPRESSED SENSING ##############################################

experiments_without_cs = {
    'with_rotation': [
        [
            'vae_convT/masked_slice/20240228-1330_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1333_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1335_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1337_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1342_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'conv_with_aux/masked_slice/20240229-1027_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240229-1029_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240229-1039_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240229-1041_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240229-1043_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['conv_enc_dec_aux/masked_slice/20240304-1656_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1701_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1703_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1706_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1708_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['deep_conv_enc_dec_aux/masked_slice/20240312-1208_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240312-1219_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240312-1224_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240319-1324_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240319-1328_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['deeper_bn_conv_enc_dec_aux/masked_slice/20240319-1411_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240319-1418_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-1215_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-1219_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240320-1553_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ]

    ],
    'without_rotation': [
        [
        'vae_convT/masked_slice/20240228-1318_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1320_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1322_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1324_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1328_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
        'conv_with_aux/masked_slice/20240229-1046_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240229-1049_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240229-1051_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240229-1053_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_with_aux/masked_slice/20240229-1107_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['conv_enc_dec_aux/masked_slice/20240304-1710_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1713_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1716_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1723_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'conv_enc_dec_aux/masked_slice/20240304-1727_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['deep_conv_enc_dec_aux/masked_slice/20240312-1221_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240312-1224_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240319-1331_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240319-1339_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deep_conv_enc_dec_aux/masked_slice/20240319-1343_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['deeper_bn_conv_enc_dec_aux/masked_slice/20240321-1254_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-1303_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-1355_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-2347_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'deeper_bn_conv_enc_dec_aux/masked_slice/20240321-2352_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ]
    ]
}







######################################### ALL DATA (includes compressed sensing) ##########################################
## 8. ALL DATA - WITH UPDATED ASCENDING AORTA - WITH ALL SUBJECTS (ANGLE CHANGE), NO ORDERING WITH ASCENDING AORTA ------------------------------------------
## Version  where we take angle 3/2 as thershold and then 1/2 if it does not work TO HAVE ALL SUBJECTS
## We do not remove the first and last 2 slices - but rather remove the two first only
## We DO NOT order the points in ascending order


# THE SEEDS HAVE BEEN FIXED - ALL DATA
experiments_with_cs = {
    'with_rotation': [
        [
            'vae_convT/masked_slice/20240218-1233_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240218-1236_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240218-1237_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240218-1240_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240218-1242_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'conv_with_aux/masked_slice/20240225-0833_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-0838_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2241_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2243_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2300_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'conv_enc_dec_aux/masked_slice/20240226-1739_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-1741_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-1746_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-1749_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-1834_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'deep_conv_with_aux/masked_slice/20240311-1626_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1632_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1637_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1646_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1648_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'deep_conv_enc_dec_aux/masked_slice/20240312-1134_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240312-1136_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240319-1350_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240319-1357_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240319-1401_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1643_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0822_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0825_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0828_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240325-0830_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ]
    ],
    'without_rotation': [
        [
            'vae_convT/masked_slice/20240225-0800_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240225-0803_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240225-0808_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240225-0811_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240225-0828_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'conv_with_aux/masked_slice/20240225-2310_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2316_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2319_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2322_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_with_aux/masked_slice/20240225-2325_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'conv_enc_dec_aux/masked_slice/20240226-0908_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-0913_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-0917_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-0920_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'conv_enc_dec_aux/masked_slice/20240226-0923_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'deep_conv_with_aux/masked_slice/20240311-1521_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1526_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1530_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1533_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_with_aux/masked_slice/20240311-1535_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'deep_conv_enc_dec_aux/masked_slice/20240312-0948_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240312-0951_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240312-0955_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240312-0958_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deep_conv_enc_dec_aux/masked_slice/20240312-1001_deep_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1311_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1314_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1318_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1321_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'deeper_bn_conv_enc_dec_aux/masked_slice/20240323-1324_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ]
    ]
}
