from utils.train_model_v6 import train_model

dropout_rate = None
make_trainable = False
gpu = True
evaluate = False


# # model_source = '../models/plain_c16x2x5_d101010_do1bc/_blank'
# model_source = '../models/_promising/plain_c16x2x5_d101010_do1bc_v2/1.767619252204895_best'
# model_dest = '../models/plain_c16x2x5_d101010_do1bc_v3'


# # model_source = '../models/plain_c16x2x5_d1040_do1bc/_blank'
# model_source = '../models/plain_c16x2x5_d1040_do1bc/3.526905502591814'
# model_dest = '../models/plain_c16x2x5_d1040_do1bc'

# model_source = '../models/plain_c16x2x5_d10404020_do1bc/_blank'
# # model_source = '../models/plain_c16x2x5_d10404020_do1bc/3.526905502591814'
# model_dest = '../models/plain_c16x2x5_d10404020_do1bc'


# model_source = '../models/plain_c16x2x5_d101010_do1bc_mod2/_blank'
# # model_source = '../models/plain_c16x2x5_d101010_do1bc_mod2/3.526905502591814'
# model_dest = '../models/plain_c16x2x5_d101010_do1bc_mod2'


# model_source = '../models/plain_c16x2x6_d1010_do3/_blank'
# # model_source = '../models/plain_c16x2x6_d1010_do3/3.526905502591814'
# model_dest = '../models/plain_c16x2x6_d1010_do3'

# model_source = '../models/plain_c20x2x6_d1212_bnorm_l2-4/_blank'
# # model_source = '../models/plain_c20x2x6_d1212_bnorm_l2-4/1.8303914319276808_temp'
# model_dest = '../models/plain_c20x2x6_d1212_bnorm_l2-4'

# model_source = '../models/plain_c16x2x6_d0502_bnorm_l2-4/_blank'
# # model_source = '../models/plain_c16x2x6_d0502_bnorm_l2-4/1.8639376523494722'
# model_dest = '../models/plain_c16x2x6_d0502_bnorm_l2-4'


# # model_source = '../models/plain_c16x2x4_d_bnorm_l2-4/_blank'
# model_source = '../models/plain_c16x2x4_d_bnorm_l2-4/2.1682458033561707'
# model_dest = '../models/plain_c16x2x4_d_bnorm_l2-4'

# # model_source = '../models/plain_c16x2x6_d1010_bnorm_do3/_blank'
# model_source = '../models/plain_c16x2x6_d1010_bnorm_do3/2.1721010208129883_best'
# model_dest = '../models/plain_c16x2x6_d1010_bnorm_do3'


# # model_source = '../models/1.9572086135546367'
# model_source = '../models/large/1.8061499746392171_temp'
# model_dest = '../models/large'


# model_source = '../models/plain_c16x2x7_d105_do1_bnorm_l2-4/_blank'
# # model_source = '../models/plain_c16x2x7_d105_do1_bnorm_l2-4/1.8061499746392171_temp'
# model_dest = '../models/plain_c16x2x7_d105_do1_bnorm_l2-4'

# # model_source = '../models/plain_c16x2x4_lc4d2_d0505_do1_bnorm_l2-4/_blank'
# model_source = '../models/plain_c16x2x4_lc4d2_d0505_do1_bnorm_l2-4/2.4585278034210205_best'
# model_dest = '../models/plain_c16x2x4_lc4d2_d0505_do1_bnorm_l2-4'

# model_source = '../models/plain_16x2x6_do3/_blank'
# model_source = '../models/currchampv1/1.795102152943611_temp'
# model_dest = '../models/currchampv1'

# # model_source = '../models/inc_c16x2_l3_d256_l1_bn_l2-4/_blank'
# model_source = '../models/inc_c16x2_l3_d256_l1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inc_c16x2_l3_d256_l1_bn_l2-4'


# model_source = '../models/inpConv_c16x2x6_l2_d1010_l1_do1_bn_l2-4/_blank'
# # model_source = '../models/inpConv_c16x2x6_l2_d1010_l1_do1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inpConv_c16x2x6_l2_d1010_l1_do1_bn_l2-4'


# model_source = '../models/inpConv_c16x2x6_skip_l2_d1010_l1_do1_bn_l2-4/_blank'
# # model_source = '../models/inpConv_c16x2x6_skip_l2_d1010_l1_do1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inpConv_c16x2x6_skip_l2_d1010_l1_do1_bn_l2-4'


# model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_do1_bn_l2-4/_blank'
# # model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_do1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inpConv_c16x2x5_skip_l2_d510_l1_do1_bn_l2-4'


# # model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_bn/_blank'
# model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_bn/1.7406492165327072_temp'
# model_dest = '../models/inpConv_c16x2x5_skip_l2_d510_l1_bn'

# # model_source = '../models/inpConv_c16x2x6_skip_l2_d510_l1_bn/_blank'
# model_source = '../models/inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7836655653715134'
# model_dest = '../models/inpConv_c16x2x6_skip_l2_d510_l1_bn'


# # model_source = '../models/inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/_blank'
# model_source = '../models/inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6982912965227106_temp'
# model_dest = '../models/inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn'


# model_source = '../models/XL_champv1'  # /_blank'
# model_source = '../models/XL_champv1/1.8429257264137269'
# model_dest = '../models/XL_champv1'


# model_source = '../models/inpConv_c16_64_256_512x3_skip_l2_d102010_l1_do15_bn/_blank'
# # model_source = '../models/inpConv_c16_64_256_512x3_skip_l2_d102010_l1_do15_bn/1.8429257264137269'
# model_dest = '../models/inpConv_c16_64_256_512x3_skip_l2_d102010_l1_do15_bn'

# # model_source = '../models/inpConv_c16_64_256_skip_d55_do25_bn/_blank'
# model_source = '../models/inpConv_c16_64_256_skip_d55_do25_bn_upToProg20Winners/2.534499648263899'
# model_dest = '../models/inpConv_c16_64_256_skip_d55_do25_bn_upToProg20Winners'

# # model_source = '../models/inpConv_c16_64_128_skip_d25_do3_bn/_blank'
# model_source = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5184312094449997_temp'
# model_dest = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3'

# # model_source = '../models/inpConv_c16_32_64_skip_d22_dobc3_bn/_blank'
# model_source = '../models/inpConv_c16_32_64_skip_d22_dobc3_bn_upToProg20Winners/2.159205729502898'
# model_dest = '../models/inpConv_c16_32_64_skip_d22_dobc3_bn_upToProg20Winners'


# model_source = '../models/inpConv_c16_64_128_skip_d25_do3_bn/_blank'
# model_source = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6231190148591996'
# model_dest = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn'

# model_source = '../models/merged_L_v1/_blank'
# # model_source = '../models/merged_L_v1/1.6231190148591996'
# model_dest = '../models/merged_L_v1'


# # model_source = '../models/merged_trained/_orig'
# model_source = '../models/merged_trained/1.6300773708820342_temp'
# model_dest = '../models/merged_trained'

# # model_source = '../models/merged_trained_progtrain/_orig'
# model_source = '../models/merged_trained_progtrain/1.624376693497823_temp'
# model_dest = '../models/merged_trained_progtrain'

# # model_source = '../models/merged_trained_progFixed/_orig'
# model_source = '../models/merged_trained_progFixed/1.6128731123209001_temp'
# model_dest = '../models/merged_trained_progFixed'

# # model_source = '../models/merged_trained_progFixed/_orig'
# model_source = '../models/merged_trained_progFixed/1.5731518268585205_best'
# model_dest = '../models/merged_mg2'
# filter = 'groups(01234);mostlyGood2'


# # model_source = '../models/merged_trained_progFixed/_orig'
# model_source = '../models/merged_mg4/1.5824660062789917_best'
# model_dest = '../models/merged_mg4'
# filter = 'groups(01234);mostlyGood4'


# model_source = '../models/merged_5models_fo/_orig'
# # model_source = '../models/merged_5models_fo/1.5824660062789917_best'
# model_dest = '../models/merged_5models_fo'
# filter = 'groups(01234);mostlyGood4'

# # model_source = '../models/merged_3models_fo/_orig'
# model_source = '../models/merged_3models_fo/1.6213901456594468_temp'
# model_dest = '../models/merged_3models_fo'
# filter = 'groups(01234);mostlyGood4'


# # model_source = '../models/5+1fold_XS/_orig'
# model_source = '../models/5+1fold_XS/1.826489714026451_temp'
# model_dest = '../models/5+1fold_XS'
# filter = 'groups(01234);mostlyGood4'

# # model_source = '../models/3fold_M/_orig'
# model_source = '../models/3fold_M/2.8491184319320477'
# model_dest = '../models/3fold_M'
# filter = 'groups(01234);mostlyGood4'


# # model_source = '../models/new/XS_c16_32_64_128_skip_d2_do1_bn/_blank'
# model_source = '../models/new/S_p1234_t1_c16_32_64_128_skip_d2_do1_bn/1.8703149026632309_temp'
# model_dest = '../models/new/S_p1234_t1_c16_32_64_128_skip_d2_do1_bn'
# filter = 'groups(1234);mostlyGood4'


# # model_source = '../models/new/XS_c16_32_64_128_skip_d2_do1_bn_nic/_blank'
# model_source = '../models/new/S_p1234_t1_c16_32_64_128_skip_d2_do1_bn_nic/2.0176358222961426_best'
# model_dest = '../models/new/S_p1234_t1_c16_32_64_128_skip_d2_do1_bn_nic'
# filter = 'groups(1234);mostlyGood4'


# # model_source = '../models/new/comb2/_orig'
# model_source = '../models/new/comb2/1.9095149995088576_temp'
# model_dest = '../models/new/comb2'
# filter = 'groups(1234);mostlyGood4'


# model_source = '../models/new/XL_p1-4_mg4/4.287527084350586_best'
# model_source = '../models/new/XL_p1-4_mg4/1.5893226511478424_temp'
# model_dest = '../models/new/XL_p1-4_mg4'
# filter = 'groups(1234);mostlyGood4'

# model_source = '../models/newTweak/from_allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/v1'
# # model_source = '../models/newTweak/v1/1.9095149995088576_temp'
# model_dest = '../models/newTweak/v1'
# filter = 'groups(1234);mostlyGood4'


# model_source = '../models/merged_XL/_orig'
# # model_source = '../models/merged_XL/1.599389974583279'
# model_dest = '../models/merged_XL'
# filter = 'groups(01234);mostlyGood4'


# model_source = '../models/merged_XL2/_orig'
# # model_source = '../models/merged_XL2/1.599389974583279'
# model_dest = '../models/merged_XL2'
# filter = 'groups(01234);mostlyGood4'


# # model_source = '../models/new/XL_op'
# model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/XL_p0_v2_do3'
# dropout_rate = 0.3
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(0);obj(result:1)(minElo:2400,result:0)(minElo:2500)'
# lr_multiplier = 0.01


# model_source = '../models/merged_XL3/_orig'
# # model_source = '../models/merged_XL3/1.5090183297395707_temp'
# model_dest = '../models/merged_XL3_train_midend'
# dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(01234);obj(result:1)(minElo:2400,result:0)(minElo:2500)'
# lr_multiplier = 0.01


# # model_source = '../models/merged_XL3/_orig'
# # model_source = '../models/merged_XL3_train_midend2/1.6687036752700806_best'
# model_source = '../models/merged_XL3_train_midend/1.628904486298561'
# model_dest = '../models/merged_XL3_train_midend2'
# dropout_rate = 0.1
# filter = 'groups(01234);obj(result:1)(willHit:1)(minElo:2400,result:0)(minElo:2500)'
# lr_multiplier = 0.001
# make_trainable = False

# model_source = '../models/new/XL_op'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/XL_p34_win_do3'
# dropout_rate = 0.3
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(34);obj(result:1)(minElo:2500,result:0)'
# lr_multiplier = 1


# model_source = '../models/merged_double/v1/1.585583065237318'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/merged_double/v1'
# dropout_rate = 0
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(01234);obj(result:1,willHit:1)(minElo:2400,result:1)(minElo:2700)'
# lr_multiplier = 0.002
# make_trainable = True

# model_source = '../models/new/XS_c16x2x4_bnorm_skip/V1/2.538818836212158_best'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/XS_c16x2x4_bnorm_skip/V3'
# dropout_rate = 0
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2400,result:1)(minElo:2700)'
# lr_multiplier = 0.3
# make_trainable = True

# model_source = '../models/new/double_kernel_M/_blank'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/double_kernel_M/V1_allButOpenings'
# dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2400,result:1)(minElo:2700)'
# lr_multiplier = 1
# make_trainable = True


# model_source = '../models/new/smallest/_blank'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/smallest/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 1
# make_trainable = True


# model_source = '../models/new/d2x8/_blank'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/d2x8/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 1
# make_trainable = True


# model_source = '../models/new/d102010/_blank'
# # model_source = '../models/new/XL_p0_v2_do2/1.5090183297395707_temp'
# model_dest = '../models/new/d102010/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 1
# make_trainable = True

# # model_source = '../models/new/c16X4x1_d10/_blank'
# model_source = '../models/new/c16X4x1_d10/V1_allButOpenings/2.1230948121547697'
# model_dest = '../models/new/c16X4x1_d10/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.02
# make_trainable = True


# # model_source = '../models/new/c16to256AndBack/_blank'
# model_source = '../models/new/c16to256AndBack/V1_allButOpenings/1.7295656563043593'
# model_dest = '../models/new/c16to256AndBack/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.3
# make_trainable = True


# # model_source = '../models/new/c16to512AndBack/_blank'
# model_source = '../models/new/c16to512AndBack/V1f_allButOpenings/1.5322560130346279_temp'
# model_dest = '../models/new/c16to512AndBack/V1f_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.01
# make_trainable = True
# gpu = True


# # model_source = '../models/new_age/v2/_blank'
# model_source = '../models/new_age/v2/p1234_v1/2.069496916770935'
# model_dest = '../models/new_age/v2/p1234_v1'
# dropout_rate = 0.1
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.3
# make_trainable = False
# gpu = True
# # evaluate = True


# # model_source = '../models/new_age/v7/_blank'
# model_source = '../models/new_age/v7/V1_allButOpenings/1.7795528173446655_best'
# model_dest = '../models/new_age/v7/V1_allButOpenings'
# # dropout_rate = 0.15
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.15
# make_trainable = True
# gpu = True
# evaluate = True


# # model_source = '../models/new_age/v8/_blank'
# model_source = '../models/new_age/v8/V1_allButOpenings/2.443315267562866_best'
# model_dest = '../models/new_age/v8/V1_allButOpenings'
# # dropout_rate = 0.15
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.5
# make_trainable = True
# gpu = True
# evaluate = True


# # model_source = '../models/new/c16to512AndBack/expanded_v4/_orig'
# model_source = '../models/new/c16to512AndBack/expanded_v4/V1_allButOpenings/1.7237210659980773_temp'
# model_dest = '../models/new/c16to512AndBack/expanded_v4/V1_allButOpenings'
# dropout_rate = 0.15
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 0.03
# make_trainable = True
# gpu = True
# evaluate = True


# model_source = '../models/new/c16-256-32_d5-10-5/_blank'
# # model_source = '../models/new/c16-256-32_d5-10-5/V1_allButOpenings/3.5583174228668213_best'
# model_dest = '../models/new/c16-256-32_d5-10-5/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 3
# make_trainable = True
# gpu = True

# model_source = '../models/new/XS_c16x2x4_bnorm_skip/_blank'
# # model_source = '../models/merged_XL3_train_midend/1.628904486298561'
# model_dest = '../models/new/XS_c16x2x4_bnorm_skip/V1'
# # dropout_rate = 0.15
# filter = 'groups(01234);obj(result:1)(willHit:1)(minElo:2400,result:0)(minElo:2500)'
# lr_multiplier = 1
# # make_trainable = True


# model_source = '../models/new_tweaked/16to256ab-Add_d5-10-5/_blank'
# # model_source = '../models/new_tweaked/16to256ab-Add_d5-10-5/V1_allButOpenings/3.5583174228668213_best'
# model_dest = '../models/new_tweaked/16to256ab-Add_d5-10-5/V1_allButOpenings'
# # dropout_rate = 0.2
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 1
# make_trainable = False
# gpu = True


# # model_source = '../models/new_age2/20_320ab/_blank'
# model_source = '../models/new_age2/20_320ab/V1_openings/1.8732330808639528'
# model_dest = '../models/new_age2/20_320ab/V1_openings'
# # dropout_rate = 0.1
# # filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(0);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2600)'
# lr_multiplier = 0.8
# make_trainable = True
# gpu = True
# # evaluate = True


# # model_source = '../models/new_age2/20_320ab/_blank'
# model_source = '../models/new_age2/20_320ab/hf_all/2.6297344118356705'
# model_dest = '../models/new_age2/20_320ab/hf_all'
# # # dropout_rate = 0.2
# # # filter = 'groups(0);obj(result:1)(minElo:2400)'
# # filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
# lr_multiplier = 2
# make_trainable = True
# gpu = True


# model_source = '../models/new/c16to512AndBack/_blank'
model_source = '../models/hf/c16to512AndBack/all/1.779046893119812_temp'
model_dest = '../models/hf/c16to512AndBack/all'
# dropout_rate = 0.2
# filter = 'groups(0);obj(result:1)(minElo:2400)'
# filter = 'groups(1234);obj(result:1,willHit:1)(minElo:2300,result:1)(minElo:2400)'
lr_multiplier = 0.2
make_trainable = True
gpu = True

initial_batch_size = 256
fixed_lr = 0.0001

train_model(model_source, model_dest,
            initial_batch_size, gpu=gpu, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr,
            dataset_reader_version='20',
            # filter='obj(percent:0.4,result:1,progressMax:0.2,opponentMinElo:2500)(percent:0.3,result:1,progressMax:0.2,opponentMinElo:2000)(percent:0.1,result:1,progressMax:0.2)',
            filter=filter,
            # filter='openings',
            ys_format='default',
            make_trainable=make_trainable,
            evaluate=evaluate,
            dropout_rate=dropout_rate,
            )

#
