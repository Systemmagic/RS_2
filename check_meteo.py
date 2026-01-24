import numpy as np
from data import EnhancedPM25Dataset
from config import Config

config = Config()
dataset = EnhancedPM25Dataset('data/CZT_PM25_2023', config)
print('气象数据（前7天）：')
for i in range(7):
    print(f'Day {i+1}: {dataset.meteo_data[i]}')
