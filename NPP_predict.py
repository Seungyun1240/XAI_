import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler


train_x, train_y = np.array([]), np.array([])
train_x_se, train_y_se = {}, {}
# want_para = ['ZINST58', 'ZINST63', 'UUPPPL', 'ZINST78', 'ZINST77', 'ZINST76', 'ZINST75', 'ZINST74', 'ZINST73',
#              'ZINST72', 'ZINST71', 'ZINST70', 'ZINST81', 'ZINST80', 'ZINST79', 'WFWLN1', 'WFWLN2', 'WFWLN3',
#              'BSDMP', 'WSTM1', 'WSTM2', 'WSTM3', 'ZINST102', 'BHV108', 'BHV208', 'BHV308', 'BTV418', 'ZINST98',
#              'ZINST15', 'KLAMPO9']

want_para = ['ZINST101', 'ZINST102', 'WSTM1', 'WSTM2', 'WSTM3', 'ZINST78', 'ZINST77', 'ZINST76', 'ZINST72',
             'ZINST71', 'ZINST70', 'ZINST75', 'ZINST74', 'ZINST73', 'ZINST58', 'ZINST56', 'UPRT', 'ZINST63',
             'KBCDO15', 'ZINST26', 'ZINST22', 'UCTMT', 'WFWLN1', 'WFWLN2', 'WFWLN3', 'BPORV', 'KLAMPO147',
             'KLAMPO148', 'KLAMPO149', 'ZINST36', 'KLAMPO9', 'KFAST', 'KLAMPO70']



scaler = MinMaxScaler()

for file in os.listdir('./DB'):
    if '.csv' in file:
        csv_db = pd.read_csv(f'./DB/{file}', index_col=0)
        get_xdb = csv_db[want_para]
        get_ydb = csv_db.loc[:, 'Normal_0'].to_numpy()
        accident_nub = {
            '12': 1,     # LOCA
            '13': 2,     # SGTR
            '15': 1,     # PORV Open (LOCA)
            '17': 1,     # Feedwater Line break (LOCA)
            '18': 3,     # Steam Line break (MSLB)
            '52': 3,     # Steam Line break - non isoable (MSLB)
        }
        get_mal_nub = file.split(',')[0][1:] # '(12, ....)' -> 12
        get_y = np.where(get_ydb != 0, accident_nub[get_mal_nub], get_ydb)
        train_x_se[file] = get_xdb
        train_y_se[file] = get_ydb
        train_x = get_xdb if train_x.shape[0] == 0 else np.concatenate((train_x, get_xdb), axis=0)
        train_y = np.append(train_y, get_y, axis=0)
        scaler.partial_fit(train_x)
        print(f'Read {file} | train_x_shape {train_x.shape} | train_y_shape {train_y.shape}')

# minmax scale
train_x = scaler.transform(train_x)
for file_ in train_x_se.keys():
    train_x_se[file_] = scaler.transform(train_x_se[file_])

save_db_info = {
    'scaler': scaler,
    'want_para': want_para,
    'train_x': train_x,
    'train_y': train_y,
    'train_x_se': train_x_se,
    'train_y_se': train_y_se,
}


with open('db_info.pkl', 'wb') as f:
    pickle.dump(save_db_info, f)




import pandas as pd
import numpy as np
import os
import pickle
import tensorflow.keras as k
from sklearn.preprocessing import MinMaxScaler
with open('db_info.pkl', 'rb') as f:
    save_db_info = pickle.load(f)

train_mode = False

model = k.Sequential([
    k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
    k.layers.Dense(300, activation='relu'),
    k.layers.Dense(150, activation='relu'),
    k.layers.Dense(200, activation='relu'),
    k.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

if train_mode:
    model.fit(save_db_info['train_x'], save_db_info['train_y'], epochs=50)
    model.save_weights('model.h5')
else:
    model.load_weights('model.h5')



import matplotlib.pyplot as plt

plt.plot(model.predict(save_db_info['train_x_se']['(52, 220600, 40).csv']))
plt.legend(['Normal', 'LOCA', 'SGTR', 'MSLB'])
plt.show()