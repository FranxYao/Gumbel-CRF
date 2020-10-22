from data_utils.dataset_ptb import DatasetPTB
from config import Config

config = Config()
config.max_sent_len = config.max_sent_len[config.dataset]
config.max_bow_len = config.max_bow_len[config.dataset]

dataset = DatasetPTB(config)
dataset.build()

num_batches = dataset.num_batches('train', 100)
batch = dataset.next_batch('train', 100)
for _ in range(num_batches): batch = dataset.next_batch('train', 100)