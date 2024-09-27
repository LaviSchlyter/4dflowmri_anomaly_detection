# Matches for different scenarios
matches_scenarios = {
  
    "Sex Equal, Age Gap <= 5": [
        ('MACDAVD_204_', 'MACDAVD_104_'),
        ('MACDAVD_308_', 'MACDAVD_101_'),
        ('MACDAVD_213', 'MACDAVD_102_'),
        ('MACDAVD_214', 'MACDAVD_107_'),
        ('MACDAVD_203_', 'MACDAVD_112_'),
        ('MACDAVD_206', 'MACDAVD_109_'),
        ('MACDAVD_305_', 'MACDAVD_106_'),
        ('MACDAVD_309_', 'MACDAVD_135'),
        ('MACDAVD_303_', 'MACDAVD_108_'),
        ('MACDAVD_307_', 'MACDAVD_131')
    ]}


##################################################################################################
##################################################################################################
# ALL DATA - balanced
##################################################################################################
##################################################################################################
experiments_with_cs = {
    'without_rotation': [[
    'simple_conv/masked_slice/20240530-1325_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'simple_conv/masked_slice/20240530-1331_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'simple_conv/masked_slice/20240530-1342_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'simple_conv/masked_slice/20240530-1344_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'simple_conv/masked_slice/20240530-1347_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3'
    ]]

}

short_experiments_with_cs = {
    'without_rotation': [[
    'simple_conv/masked_slice/20240530-1325_simple_conv_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3'
    ]]

}

"""


short_experiments_with_cs = {
    'without_rotation': [
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240520-2008_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240520-2010_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3']
    ]
    }


short_experiments_with_cs = {
    'without_rotation': [
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240521-1834_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240521-1837_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240521-1839_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240521-1853_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240521-1912_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3']
    ]

}



short_experiments_with_cs = {
    'with_rotation': [[]],
    'without_rotation': [[
    'vae_convT/masked_slice/20240518-2136_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20240518-2145_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20240518-2147_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20240518-2152_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'vae_convT/masked_slice/20240518-2155_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3'
    ]]

}

short_experiments_with_cs = {
    'without_rotation': [
    ['deeper_conv_enc_dec_aux/masked_slice/20240524-1037_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_conv_enc_dec_aux/masked_slice/20240524-1041_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_conv_enc_dec_aux/masked_slice/20240524-1133_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_conv_enc_dec_aux/masked_slice/20240524-1143_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_conv_enc_dec_aux/masked_slice/20240524-1243_deeper_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3']
    ]

}
short_experiments_with_cs = {
    'without_rotation': [
    ['deep_conv_with_aux/masked_slice/20240520-1909_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deep_conv_with_aux/masked_slice/20240520-1911_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deep_conv_with_aux/masked_slice/20240520-2013_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deep_conv_with_aux/masked_slice/20240520-2015_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deep_conv_with_aux/masked_slice/20240520-2020_deep_conv_with_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3']
    ]

}
short_experiments_with_cs = {
    'without_rotation': [
    ['deeper_bn_conv_enc_dec_aux/masked_slice/20240519-2101_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240519-2106_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240519-2109_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240519-2112_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
    'deeper_bn_conv_enc_dec_aux/masked_slice/20240519-2115_deeper_bn_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3'
    ]]

}
"""







##################################################################################################
##################################################################################################
# RECONSTRUCTION BASED
##################################################################################################
##################################################################################################



experiments_only_cs_reconstruction_based = {
    'with_rotation': [
        [
            'vae_convT/masked_slice/20240503-1212_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240503-1305_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240503-1317_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240503-1319_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240503-1805_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'vae_convT/masked_slice/20240425-1618_vae_convT_masked_slice_lr1.000e-03-e2000-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240425-1620_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240515-1624_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240515-1629_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240515-1724_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__with_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3'
            
        ]
    ],
    'without_rotation': [
        [
            'vae_convT/masked_slice/20240503-1209_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1306_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1312_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1313_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1323_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [
            'vae_convT/masked_slice/20240425-1634_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240425-1623_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240515-1627_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240515-1631_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240515-1721_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3'
            
        ]
    ]
}

experiments_without_cs_reconstruction_based = {
    'with_rotation': [
        [
            'vae_convT/masked_slice/20240506-1325_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1329_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1410_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1411_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-1413_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        []
    ],
    'without_rotation': [
        [
            'vae_convT/masked_slice/20240506-1421_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-2137_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-2139_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-2140_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240506-2141_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['vae_convT/masked_slice/20240515-1735_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240515-1736_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240515-1737_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240515-1817_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240515-1828_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_unwrapped_decreased_interpolation_factor_cube_3']
    ]
}


experiments_with_cs_reconstruction_based = {
    'without_rotation': [
        [
            'vae_convT/masked_slice/20240526-1135_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_5_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240526-1142_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_10_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240526-1209_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_15_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240526-1217_vae_convT_masked_slice_lr1.000e-03-e1500-bs8-gf_dim8-daFalse-f100__SEED_20_2Dslice__without_rotation_with_cs_skip_updated_ao_S10_balanced_decreased_interpolation_factor_cube_3'
            
        ]
    ]
}







# ==================================================================================================
# ==================================================================================================
# ==================================================================================================

# Not used for paper

##################################################################################################
##################################################################################################
# ONLY COMPRESSED SENSING
##################################################################################################
##################################################################################################
short_experiments_only_cs = {
    'without_rotation': [[
        'vae_convT/masked_slice/20240215-0858_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0901_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0906_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0910_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0914_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ]]
}

experiments_only_cs = {
    'without_rotation': [[
        'vae_convT/masked_slice/20240215-0858_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0901_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0906_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0910_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240215-0914_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
    ],
    ['vae_convT/masked_slice/20240508-1415_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240508-1421_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240508-1424_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240508-1425_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240508-1452_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
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
        ],
        ['conv_enc_dec_aux/masked_slice/20240509-1053_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
         'conv_enc_dec_aux/masked_slice/20240509-1056_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
         'conv_enc_dec_aux/masked_slice/20240509-1057_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
         'conv_enc_dec_aux/masked_slice/20240509-1100_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
         'conv_enc_dec_aux/masked_slice/20240509-1101_conv_enc_dec_aux_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_only_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3']

    ]

}





##################################################################################################
##################################################################################################
# ONLY NON COMPRESSED SENSING
##################################################################################################
##################################################################################################


short_experiments_without_cs = {
    'with_rotation': [[]],
    'without_rotation': [['vae_convT/masked_slice/20240511-0918_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1032_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1143_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1150_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1152_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3'
            ]]

}



experiments_without_cs = {
    'with_rotation': [
        [
            'vae_convT/masked_slice/20240228-1330_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1333_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1335_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1337_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240228-1342_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__with_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        [],
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
        ],
        [], # centered_norm_scheduler
        [] # centered_norm

    ],
    'without_rotation': [
        [
        'vae_convT/masked_slice/20240228-1318_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1320_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1322_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1324_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240228-1328_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_poisson_mix_training_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
        ],
        ['vae_convT/masked_slice/20240507-1332_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240507-1341_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240507-1343_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240507-1345_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3',
        'vae_convT/masked_slice/20240507-1352_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_decreased_interpolation_factor_cube_3'
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
        ],
        ['vae_convT/masked_slice/20240510-1136_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240510-1139_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240510-1142_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240510-1213_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
         'vae_convT/masked_slice/20240511-0915_vae_convT_masked_slice_SSL_lr1.000e-03_scheduler-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3'
         ],
         ['vae_convT/masked_slice/20240511-0918_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_5_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1032_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_10_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1143_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_15_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1150_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_20_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3',
            'vae_convT/masked_slice/20240511-1152_vae_convT_masked_slice_SSL_lr1.000e-03-e1500-bs8-gf_dim8-daFalse__SEED_25_2Dslice__without_rotation_without_cs_skip_updated_ao_S10_centered_norm_decreased_interpolation_factor_cube_3'
            ]

    ]
}







