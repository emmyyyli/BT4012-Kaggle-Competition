"""Stage-2 feature list for the pruned variant (report/findings.md section 45).

The 151 stage-2 features that together carry 95% of mean |TreeSHAP| on test rows of the final
model (seed 0, fitted on all labeled rows; ``artifacts/report/final_model_feature_importance.csv``).
Selected from test-row features only; no test labels are used. Ordered by influence.
"""

STAGE2_KEEP_95 = [
    'feat_125', 'hop2_score_mean', 'feat_61', 'feat_55', 'feat_53', 'feat_16',
    'feat_5', 'feat_3', 'feat_136', 'feat_161', 'feat_163', 'feat_58',
    'feat_67', 'feat_52', 'feat_132', 'hop2_mean_feat_3', 'feat_90', 'nbr_score_mean',
    'hop2_mean_feat_90', 'feat_131', 'feat_59', 'nbrw_max_feat_90', 'feat_101', 'nbr_score_max',
    'feat_65', 'feat_14', 'nbrw_mean_feat_2', 'feat_41', 'feat_2', 'feat_9',
    'nbrw_max_feat_53', 'feat_43', 'feat_47', 'feat_49', 'nbrw_mean_feat_90', 'feat_8',
    'feat_159', 'feat_76', 'feat_29', 'nbrw_mean_feat_59', 'nbrw_out_mean_feat_106', 'nbrw_std_feat_125',
    'feat_130', 'nbrw_max_feat_65', 'feat_139', 'feat_46', 'feat_103', 'nbrw_std_feat_2',
    'feat_23', 'feat_89', 'feat_158', 'feat_138', 'nbrw_max_feat_59', 'nbrw_in_mean_feat_90',
    'feat_10', 'feat_102', 'feat_88', 'feat_18', 'feat_1', 'feat_80',
    'feat_106', 'nbrw_out_mean_feat_161', 'feat_127', 'hop2_mean_feat_161', 'feat_66', 'nbrw_mean_feat_23',
    'depth_up', 'feat_64', 'nbrw_mean_feat_16', 'hop2_mean_feat_163', 'feat_137', 'hop2_mean_feat_131',
    'feat_13', 'feat_107', 'feat_165', 'nbrw_out_mean_feat_90', 'feat_164', 'nbr_score_mean_out',
    'feat_97', 'hop2_mean_feat_2', 'feat_17', 'feat_162', 'nbr_score_mean_in', 'feat_119',
    'feat_100', 'hop2_mean_feat_106', 'feat_79', 'feat_109', 'depth_down', 'feat_154',
    'feat_121', 'nbrw_mean_feat_65', 'feat_25', 'feat_20', 'feat_142', 'feat_60',
    'feat_81', 'feat_72', 'nbrw_out_mean_feat_52', 'feat_40', 'nbrw_std_feat_8', 'nbrw_mean_feat_125',
    'feat_148', 'feat_155', 'feat_31', 'nbrw_std_feat_47', 'feat_157', 'frac_labeled_nbrs',
    'feat_151', 'nbrw_out_mean_feat_47', 'feat_77', 'nbrw_out_mean_feat_2', 'feat_35', 'nbrw_max_feat_16',
    'nbrw_out_mean_feat_55', 'nbrw_mean_feat_3', 'nbrw_in_mean_feat_16', 'nbrw_std_feat_136', 'nbrw_out_mean_feat_53', 'nbrw_mean_feat_58',
    'feat_156', 'feat_22', 'feat_11', 'nbrw_max_feat_55', 'nbrw_max_feat_67', 'hop2_mean_feat_125',
    'nbrw_mean_feat_161', 'feat_54', 'feat_95', 'nbrw_mean_feat_132', 'nbrw_in_mean_feat_59', 'nbrw_max_feat_2',
    'feat_21', 'feat_114', 'feat_91', 'feat_4', 'nbrw_max_feat_8', 'feat_6',
    'feat_19', 'feat_160', 'nbrw_out_mean_feat_61', 'hop2_mean_feat_136', 'hop2_labeled_walks', 'nbrw_max_feat_61',
    'feat_78', 'feat_30', 'nbrw_mean_feat_53', 'nbrw_out_mean_feat_163', 'hop2_mean_feat_5', 'hop2_mean_feat_65',
    'feat_120',
]
