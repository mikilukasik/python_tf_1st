from utils import load_model, prepare_model_for_transfer_learning

model = load_model(
    './models/c4RESd2_S_v1/1.6770610121091205')
model.summary()

model = prepare_model_for_transfer_learning(model, [5000, 3000])

model.summary()
