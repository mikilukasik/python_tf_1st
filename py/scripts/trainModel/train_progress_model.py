from utils.train_model_v6 import train_model

# model_source = '../models/c16_32_64_skip_d22_dobc3_bn_o5/_blank'
model_source = '../models/c16_32_64_skip_d22_dobc3_bn_o5/0.8450775146484375_best'
model_dest = '../models/c16_32_64_skip_d22_dobc3_bn_o5'
# # filter = 'obj(minElo:2600)(result:1,minElo:2400)(willHit:1,minElo:2400)(percent:0.3,willHit:1)(percent:0.1,result:1)'
# filter = 'groups(01234)'

# model_source = '../models/c16_32_64_skip_d22_dobc3_bn_o1/_blank'
# model_source = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v2/0.04929423820599913'
# model_dest = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v3'

# model_source = '../models/c16x2x5_skip_d52_do2_bn_o5/_blank'
# # model_source = '../models/c16x2x5_skip_d52_do2_bn_o5/0.04929423820599913'
# model_dest = '../models/c16x2x5_skip_d52_do2_bn_o5'


# filter = 'obj(minElo:2600)(result:1,minElo:2400)(willHit:1,minElo:2400)(percent:0.3,willHit:1)(percent:0.1,result:1)'
# filter = 'groups(01234);obj(progressMin:0.24)(progressMax:0.16)'
filter = 'groups(01234);almostall'

initial_batch_size = 160
lr_multiplier = 0.3  # 0.01  # 0.003  # 0.1
fixed_lr = 0.0002

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr,
            dataset_reader_version='20',
            # fresh_reader=True,
            # filter='obj(percent:0.4,result:1,progressMax:0.2,opponentMinElo:2500)(percent:0.3,result:1,progressMax:0.2,opponentMinElo:2000)(percent:0.1,result:1,progressMax:0.2)',
            filter=filter,
            # filter='openings',
            ys_format='progressGroup5', make_trainable=True, evaluate=True)

#
