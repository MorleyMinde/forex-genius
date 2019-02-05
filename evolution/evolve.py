from devol import DEvol
from genome_handler import GenomeHandler

import pandas as pd
dataset = pd.read_csv('/home/vincent/gym-trader/data/EURUSD/raw/EURUSD_Candlestick_1_M_2014labeledtr.csv')
dataset = dataset.dropna()
print(dataset.Volume)
dataset = dataset[dataset.Volume != 0]
X = dataset[['Open', 'High', 'Low', 'Close']]
Y = dataset[['Action']]

split = int(len(dataset)*0.9)
X_train, X_test, y_train, y_test = X[:split], X[split:], Y[:split], Y[split:]
dataset = ((X_train, y_train), (X_test, y_test))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.
print(X_train.shape[:-1])
genome_handler = GenomeHandler(max_conv_layers=6,
                               max_dense_layers=2, # includes final dense layer
                               max_filters=256,
                               max_dense_nodes=1024,
                               input_shape=X_train.shape[1:],
                               n_classes=10)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.
# The best model is returned decoded and with `epochs` training done.

devol = DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=20,
                  pop_size=20,
                  epochs=5)
print(model.summary())
