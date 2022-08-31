# %% imports we know we'll need
from __future__ import print_function
from scipy import special
import numpy as np
import tensorflow as tf
import BER_calc
import scipy.io as sio
from sklearn.utils import shuffle
import os
import random
import matlab
import matlab.engine
import py
import numpy as np
eng = matlab.engine.start_matlab()
from matlab._internal.mlarray_sequence  import _MLArrayMetaClass
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
import h5py




def create_dataset_m(x_pol_in_complex, y_pol_in_complex, x_pol_des_complex, n_sym1, n_symout1):
    raw_size = x_pol_in_complex.shape[0]
    dataset_size = raw_size - 2 * n_sym1
    dataset_range = n_sym1 + np.arange(dataset_size)

    dataset_x_pol__des = np.empty([dataset_size, n_symout1, 2], dtype='float64')
    dataset_x_pol__des[:] = np.nan

    # dataset_x_pol__des[:, 0] = np.real(x_pol_des_complex[dataset_range])
    # dataset_x_pol__des[:, 1] = np.imag(x_pol_des_complex[dataset_range])

    dataset_x_pol__in = np.empty([dataset_size, n_sym1, 4], dtype='float64')
    dataset_x_pol__in[:] = np.nan

    bnd_vec = int(np.floor(n_sym1 / 2))
    bnd_vec_out = int(np.floor(n_symout1 / 2))
    for vec_idx, center_vec in enumerate(dataset_range):
        local_range = center_vec + np.arange(-bnd_vec, bnd_vec + 1)
        local_range_out = center_vec + np.arange(-bnd_vec_out, bnd_vec_out + 1)
        n1 = np.arange(0, n_sym1)
        n2 = np.arange(0, n_symout1)
        # n2 = np.arange(n_sym, 2 * n_sym)
        # n3 = np.arange(2 * n_sym, 3 * n_sym)
        # n4 = np.arange(3 * n_sym, 4 * n_sym)
        if np.any(local_range < 0) or np.any(local_range > raw_size):
            ValueError('Local range steps out of the data range during dataset creation!!!')
        else:
            dataset_x_pol__in[vec_idx, n1, 0] = np.real(x_pol_in_complex[local_range])
            dataset_x_pol__in[vec_idx, n1, 1] = np.imag(x_pol_in_complex[local_range])
            dataset_x_pol__in[vec_idx, n1, 2] = np.real(y_pol_in_complex[local_range])
            dataset_x_pol__in[vec_idx, n1, 3] = np.imag(y_pol_in_complex[local_range])
            dataset_x_pol__des[vec_idx, n2, 0] = np.real(x_pol_des_complex[local_range_out])
            dataset_x_pol__des[vec_idx, n2, 1] = np.imag(x_pol_des_complex[local_range_out])

    if np.any(np.isnan(dataset_x_pol__in)):
        ValueError('Dataset matrix wasn''t fully filled by data!!!')
    dataset_x_pol__in, dataset_x_pol__des = shuffle(dataset_x_pol__in, dataset_x_pol__des)
    return dataset_x_pol__in, dataset_x_pol__des



def BER_est(x_in, x_ref):
    QAM_order = QAM
    return BER_calc.QAM_BER_gray(x_in, x_ref, np.array(QAM_order))


def BER_Calculation_multisymbol(x_in, x_ref):
    n_symbols_recovered = len(x_ref[0, :, 0])
    n_symbols_transmitted = len(x_in[0, :, 0])
    n_sequences = len(x_ref[:, 0, 0])
    bnd_vec_out = int(np.floor(n_symbols_recovered / 2))
    range_input = int(n_symbols_transmitted / 2) + + np.arange(-bnd_vec_out, bnd_vec_out + 1)
    BER_sum = 0
    for i in range(n_sequences):
        x_trans_complex = x_ref[i, :, 0] + 1j * x_ref[i, :, 1]
        x_received_complex = x_in[i, range_input, 0] + 1j * x_in[i, range_input, 1]
        BER_sum = BER_sum + BER_est(x_received_complex, x_trans_complex)
    BER_total = BER_sum / n_sequences
    return BER_total


def BER_Calculation_multisymbol2(x_in, x_ref):
    n_symbols_recovered = len(x_ref[0, :, 0])
    n_symbols_transmitted = len(x_in[0, :, 0])
    n_sequences = len(x_ref[:, 0, 0])
    bnd_vec_out = int(np.floor(n_symbols_recovered / 2))
    range_input = int(n_symbols_transmitted / 2) + np.arange(-bnd_vec_out, bnd_vec_out + 1)
    BER_sum = 0
    for i in range(int(n_sequences / n_symbols_recovered)):
        x_trans_complex = x_ref[i * n_symbols_recovered, :, 0] + 1j * x_ref[i * n_symbols_recovered, :, 1]
        x_received_complex = x_in[i * n_symbols_recovered, range_input, 0] + 1j * x_in[
            i * n_symbols_recovered, range_input, 1]
        BER_sum = BER_sum + BER_est(x_received_complex, x_trans_complex)
    BER_total = BER_sum / int(n_sequences / n_symbols_recovered)
    return BER_total


def BER_Calculation_multisymbol3(x_in, x_ref):
    n_symbols_recovered = len(x_ref[0, :, 0])
    n_sequences = len(x_ref[:, 0, 0])
    BER_sum = 0
    for i in range(int(n_sequences / n_symbols_recovered)):
        x_trans_complex = x_ref[i * n_symbols_recovered, :, 0] + 1j * x_ref[i * n_symbols_recovered, :, 1]
        x_received_complex = x_in[i * n_symbols_recovered, :, 0] + 1j * x_in[
                                                                        i * n_symbols_recovered, :, 1]
        BER_sum = BER_sum + BER_est(x_received_complex, x_trans_complex)
    BER_total = BER_sum / int(n_sequences / n_symbols_recovered)
    return BER_total


QAM = 64
n_taps = 110
n_taps_out = 97
n_sym = 2 * n_taps + 1
n_sym_out = 2 * n_taps_out + 1
typy_activationp = 'linear'
lr1 = 0.0005 #learning rate
training_epochs = 200000
batchs = 2000


size_kernel = tuple([int(n_sym - n_sym_out + 1)])

MATLAB_file_name = 'Dataset_2dBm_64QAM_expPS_.mat'
Dataset = h5py.File(MATLAB_file_name)



dataset_x_pol_in_test, dataset_x_pol_des_test = np.asarray(Dataset['test_in']), np.asarray(Dataset['test_des'])



dataset_x_pol_in_test, dataset_x_pol_des_test = np.transpose(dataset_x_pol_in_test,
(2, 1, 0)), np.transpose(dataset_x_pol_des_test,
(2, 1, 0))

n_train = np.ceil((2 ** 18) / batchs)
n_test = np.floor(len(dataset_x_pol_in_test) / batchs)
range_train = np.arange(int(batchs * n_train))
range_test = np.arange(int(batchs * n_test))
#####################################################################################################

inputs = tf.keras.Input(shape=(n_sym, 4), name='digits')
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(35, return_sequences=True))(inputs)
outputs = tf.keras.layers.Conv1D(2, size_kernel, activation=typy_activationp)(x)
model_w_aug = tf.keras.Model(inputs=inputs, outputs=outputs)
model_w_aug.compile(loss=tf.keras.losses.mean_squared_error,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr1),  # keras.optimizers.Adam(lr=0.001),
                    metrics=['accuracy'])

model_w_aug.summary()


berbest = 10000
BER_test = np.zeros(training_epochs)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# Wrap data in Dataset objects.
test_data = tf.data.Dataset.from_tensor_slices((
    dataset_x_pol_in_test[range_test]))

# The batch size must now be set on the Dataset objects.
test_data = test_data.batch(batchs)

# Disable AutoShard.
test_data = test_data.with_options(options)
Best_BER_no_NN = BER_Calculation_multisymbol2(dataset_x_pol_in_test[range_test], dataset_x_pol_des_test[range_test])


for epoch in range(training_epochs):
    guard_band = 1000
    print('Start Training')
    beta_2_random = random.uniform(10,20)
    gamma_random = random.uniform(1,2)
    alpha_random = random.uniform(0.1,0.3)
    roll_off_random = random.uniform (0.01,0.2)
    NF_random = random.uniform (3,6)
    fiber_length= 110
    n_span = 9
    power = 2
    symbol_rate = 34

    [Out_raw_complex_train, In_raw_complex_train, Out_raw_complexy_train, In_raw_complexy_train, Out_raw_complex_test,
     In_raw_complex_test, Out_raw_complexy_test, In_raw_complexy_test] = (
        eng.Sim_data_func(QAM, fiber_length, n_span, power, symbol_rate, 4*guard_band+len(range_train), beta_2_random, gamma_random, alpha_random , roll_off_random, NF_random, nargout=8))
    print('Finished new data')
    length_raw_complex_train = int(len(In_raw_complex_train))
    length_raw_complex_test = int(len(In_raw_complex_test))
    train_range = range(guard_band, int(length_raw_complex_train - guard_band))
    test_range = range(guard_band, int(length_raw_complex_test - guard_band))

    #training
    x_pol_in_raw_complex_train = In_raw_complex_train[train_range]
    y_pol_in_raw_complex_train = In_raw_complexy_train[train_range]
    x_pol_des_raw_complex_train = Out_raw_complex_train[train_range]
    y_pol_des_raw_complex_train = Out_raw_complexy_train[train_range]
    dataset_x_pol_in_train, dataset_x_pol_des_train = create_dataset_m(x_pol_in_raw_complex_train,
                                                                       y_pol_in_raw_complex_train,
                                                                       x_pol_des_raw_complex_train,
                                                                       n_sym, n_sym_out)

    dataset_x_pol_in_train, dataset_x_pol_des_train = shuffle(dataset_x_pol_in_train, dataset_x_pol_des_train)

    #testing
    x_pol_in_raw_complex_test = In_raw_complex_train[test_range]
    y_pol_in_raw_complex_test = In_raw_complexy_train[test_range]
    x_pol_des_raw_complex_test = Out_raw_complex_train[test_range]
    y_pol_des_raw_complex_test = Out_raw_complexy_train[test_range]
    dataset_x_pol_in_test, dataset_x_pol_des_test = create_dataset_m(x_pol_in_raw_complex_test,
                                                                       y_pol_in_raw_complex_test,
                                                                       x_pol_des_raw_complex_test,
                                                                       n_sym, n_sym_out)


    dataset_x_pol_in_test, dataset_x_pol_des_test = shuffle(dataset_x_pol_in_test, dataset_x_pol_des_test)
    # Wrap data in Dataset objects.
    train_data = tf.data.Dataset.from_tensor_slices((
        dataset_x_pol_in_train[range_train],
        dataset_x_pol_des_train[range_train]))

    # test_data = tf.data.Dataset.from_tensor_slices((
    #     dataset_x_pol_in_test[range_test],
    #     dataset_x_pol_des_train[range_test]))

    # The batch size must now be set on the Dataset objects.
    train_data = train_data.batch(batchs)
    train_data = train_data.with_options(options)

    # test_data = test_data.batch(batchs)
    # test_data = test_data.with_options(options)

    model_w_aug.fit(train_data, epochs=1, verbose=1)
    # model_w_aug.fit(test_data, epochs=1, verbose=0)

    # test_data = tf.data.Dataset

    Predriction_test = model_w_aug.predict(test_data)

    Predriction_train = model_w_aug.predict(train_data)

    BER_test[epoch] = BER_Calculation_multisymbol3(Predriction_test, dataset_x_pol_des_test[range_test])

    BER_train = BER_Calculation_multisymbol3(Predriction_train, dataset_x_pol_des_train[range_train])
    BER_train_ref = BER_Calculation_multisymbol2(dataset_x_pol_in_train[range_train], dataset_x_pol_des_train[range_train])

    if berbest >= BER_test[epoch]:
        berbest = BER_test[epoch]
        Best_Epoch_NN = epoch
    Q_prop_ref2 = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * berbest))
    Q_prop_ref3 = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER_test[epoch]))
    Q_prop_ref4 = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * Best_BER_no_NN))

    Q_train_predict = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER_train))
    Q_train_ref = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER_train_ref))

    optt = (Best_BER_no_NN - berbest) * 100 / Best_BER_no_NN
    optt2 = (Best_BER_no_NN - BER_test[epoch]) * 100 / Best_BER_no_NN

    print('Epoch now', epoch, 'Epoch best', Best_Epoch_NN, 'Gain best', optt, 'Gain now', optt2)
    print('Q_train Sim now', Q_train_predict,'Q_train Sim ref now', Q_train_ref, 'Q_test Exp now', Q_prop_ref3, 'Q-factor Exp NN best',  Q_prop_ref2, 'Q-factor  Exp ref', Q_prop_ref4)