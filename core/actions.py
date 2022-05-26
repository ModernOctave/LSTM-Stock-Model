import math
from core.data_processor import DataLoader
from core.model import Model
from core.plot import plot_results, plot_results_multiple


def make_new_model(configs, data, inmemory=False):
    model = Model()
    model.build_model(configs)

    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
    if inmemory:
        # in-memory training
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
    else:
        # out-of memory generative training
        steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )

    return model

def run_predict(model: Model, data: DataLoader, configs):
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['prediction']['length'])
    model.evaluate(x_test, y_test)

    plot_results_multiple(predictions, y_test, configs['prediction']['length'])

def train_further(model: Model, configs, data: DataLoader, inmemory=False):
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
    if inmemory:
        # in-memory training
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
    else:
        # out-of memory generative training
        steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )