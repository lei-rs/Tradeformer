from tqdm import trange
import pandas as pd
import numpy as np


def generate_data():
    returns = pd.read_csv('synth_component_returns.csv', index_col=0).sort_index(axis=1)

    total_stocks = 10000
    synth_data = [0] * total_stocks
    synth_weights = pd.DataFrame(np.zeros((total_stocks, returns.shape[1])))
    synth_weights.columns = returns.columns
    for i in trange(total_stocks):
        num_stocks = np.random.randint(100, 600)
        portfolio = sorted(np.random.choice(returns.columns, num_stocks, replace=False))
        weights = np.random.dirichlet(np.ones(num_stocks))
        synth_weights.loc[i, portfolio] = weights
        synth_return = np.sum(returns[portfolio] * weights, axis=1)
        synth_data[i] = synth_return

    synth_data = pd.DataFrame(synth_data).T.astype(np.float32)
    synth_data.columns = synth_data.columns.astype(str)
    synth_weights = pd.DataFrame(synth_weights)
    synth_data.to_parquet('synth_data.parquet')
    synth_weights.to_csv('synth_weights.csv')


if __name__ == '__main__':
    generate_data()
