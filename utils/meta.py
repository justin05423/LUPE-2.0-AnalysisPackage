# data_root_dir = '/Volumes/data/lupe'
data_root_dir = '/Users/justinjames/LUPE_Corder-Lab'

###### DO NOT CHANGE ANYTHING BELOW ######
behavior_names = ['still',
                  'walking',
                  'rearing',
                  'grooming',
                  'licking hindpaw L',
                  'licking hindpaw R']

behavior_colors = ['crimson',
                   'darkcyan',
                   'goldenrod',
                   'royalblue',
                   'rebeccapurple',
                   'mediumorchid']

keypoints = ["nose", "mouth", "l_forepaw", "l_forepaw_digit", "r_forepaw", "r_forepaw_digit",
             "l_hindpaw", "l_hindpaw_digit1", "l_hindpaw_digit2", "l_hindpaw_digit3",
             "l_hindpaw_digit4", "l_hindpaw_digit5", "r_hindpaw", "r_hindpaw_digit1",
             "r_hindpaw_digit2", "r_hindpaw_digit3", "r_hindpaw_digit4", "r_hindpaw_digit5",
             "genitalia", "tail_base"]

# 30 pixels = 1 cm
pixel_cm = 0.0330828

sexes = ['Male', 'Female']
drugs_study5 = ['Morphine', 'Morphine_Formalin']

groups = [f'Group{i}' for i in range(1, 8)]
groups_sni = ['A_Baseline_NoSNI', 'B_Baseline_SNI', 'C_3ISNI_DCZ', 'D_4WSNI_DCZ']
groups_raquel = ['D1', 'D3', 'D7', 'D14', 'D21']
groups_study4 = ['Male', 'Female', 'Combined']
groups_study5 = ['Male', 'Female', 'Combined']
groups_study6 = ['Male', 'Female', 'Combined']
groups_sni_sex = ['A_Baseline_NoSNI', 'B_Baseline_SNI', 'C_3WSNI_DCZ', 'D_4WSNI_DCZ']
groups_sni_combined = ['A_Baseline_NoSNI', 'B_Baseline_SNI', 'C_3WSNI_DCZ', 'D_4WSNI_DCZ']
groups_lmw_mor_reexpression = ['Male', 'Female', 'Combined']
groups_lle_rILN = ['Group1_SNI','Group2_Uninjured-Formalin']
groups_cso_sar_miniscope = ['M1','M3']

conditions = [f'Condition{i}' for i in range(1, 4)]
conditions_extra = ['Condition3A', 'Condition3B']
conditions_hab = ['HAB_D1_Male', 'HAB_D1_Female', 'HAB_D2_Male', 'HAB_D2_Female']
conditions_raquel = ['Control', 'Experimental']
conditions_study4 = ['Control', 'Experimental']
conditions_study5 = ['Control', 'Experimental']
conditions_study6 = ['Control', 'Experimental']
conditions_sni_sex = ['control_mMORP-eYFP_MALE', 'control_mMORP-eYFP_FEMALE', 'exp_mMORP-hm4di_MALE', 'exp_mMORP-hm4di_FEMALE']
conditions_sni_combined = ['control_mMORP-eYFP', 'exp_mMORP-hm4di']
conditions_lmw_mor_reexpression = ['Control', 'Exp_MOR-ko']
conditions_lle_rILN = ['rILN_eYFP','rILN_hm4Di']
conditions_cso_sar_miniscope = ['baseline','cap','morphine','morphine-capsaicin']

groups_noci_morp_dreadd = ['Combined', 'Female', 'Male']
conditions_noci_morp_dreadd = ['EXP_CONFON', 'EXP_MORP', 'EXP_YFP']

groups_bla_sni_opto_inhibit = ['SNI', 'NoSNI']
conditions_bla_sni_opto_inhibit = ['CTRL_FLEX', 'EXP_stGtACR2_inhibit']
groups_project_cso_capsaicin2ug = ['M1', 'M2', 'M3', 'M4']
conditions_project_cso_capsaicin2ug = ['baseline', 'test_2ug']
