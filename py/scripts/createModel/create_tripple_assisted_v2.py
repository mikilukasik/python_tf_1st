
from keras.models import Model
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, Input
from keras.initializers import he_normal
from helpers.create_champ_model import create_champ_model
from utils.save_model import save_model

model_folder = '../models/tripple_assisted_v2/_blank'

input = Input(shape=(8, 8, 14))
flat_input = Flatten(name='flat_input')(input)


from_block_dense = create_champ_model(
    input=input, flat_input=flat_input, return_dense_index=0, returned_dense_name='from_block_dense', layer_name_prefix='from_block')
to_block_dense = create_champ_model(
    input=input, flat_input=flat_input, return_dense_index=0, returned_dense_name='to_block_dense', layer_name_prefix='to_block')
shared_block_dense = create_champ_model(
    input=input, flat_input=flat_input, return_dense_index=0, returned_dense_name='shared_block_dense', layer_name_prefix='shared_block')

combined_feed_concat = Concatenate()(
    [flat_input, from_block_dense, shared_block_dense, to_block_dense])
combined_dense = Dense(768, ELU(), name='combined-dense')(combined_feed_concat)

from_feed_concat = Concatenate()(
    [flat_input, from_block_dense,  shared_block_dense])
from_dense = Dense(512, ELU(), name='from-dense')(from_feed_concat)

to_feed_concat = Concatenate()(
    [flat_input, shared_block_dense, to_block_dense])
to_dense = Dense(512, ELU(), name='to-dense')(to_feed_concat)


combined_softmax = Dense(1837, 'softmax', name='combined-softmax')(
    Concatenate()([flat_input, from_dense, combined_dense, to_dense]))
from_softmax = Dense(64, 'softmax', name='from-softmax')(Concatenate()
                                                         ([flat_input, from_dense, combined_dense]))

to_concat = Concatenate()([flat_input, combined_dense, to_dense])

to_softmax = Dense(64, 'softmax', name='to-softmax')(to_concat)
knight_promo = Dense(1, 'sigmoid', name='knight-promo')(to_concat)

output = Concatenate(name='concat-output')([
    combined_softmax,
    from_softmax,
    to_softmax,
    knight_promo
])


model = Model(inputs=input, outputs=output)

model.summary()
save_model(model, model_folder)
