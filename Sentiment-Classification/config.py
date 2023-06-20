import os
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--seed", type=int, default=3847)
args.add_argument("--batch_size", type=int, default=512)
args.add_argument("--epoch", type=int, default=20)
args.add_argument("--output_path", type=str, default="./output")
args.add_argument("--test", type=bool, default=False)
args.add_argument("--learning_rate", type=float, default=1e-2)
args.add_argument("--seq_len", type=int, default=120)
args.add_argument("--hidden_size", type=int, default=16)
args.add_argument("--dropout", type=float, default=0.8)
args.add_argument("--model_path", type=str, default="./output/model.pth")
args.add_argument(
    "--model_type",
    type=str,
    default="LSTM",
    choices=["LSTM", "GRU", "CNN"],
)
args = args.parse_args()

DATASET_PATH = './Dataset'
TRAIN_SET_PATH = os.path.join(DATASET_PATH, 'train.txt')
VALIDATION_SET_PATH = os.path.join(DATASET_PATH, 'validation.txt')
TEST_SET_PATH = os.path.join(DATASET_PATH, 'test.txt')
WORD2VEC_PATH = os.path.join(DATASET_PATH, 'wiki_word2vec_50.bin')

BATCH_SIZE = args.batch_size
EPOCH = args.epoch
IS_TEST = args.test
LEARNING_RATE = args.learning_rate
MODEL_TYPE = args.model_type
SEQ_LEN = args.seq_len
HIDDEN_SIZE = args.hidden_size
DROPOUT = args.dropout

OUTPUT_PATH = args.output_path
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
MODEL_PATH = args.model_path
LOG_PATH = os.path.join(OUTPUT_PATH, 'log.txt')
FIGURE_PATH = os.path.join(OUTPUT_PATH, 'figure.png')
EMBEDDING_MATRIX_PATH = os.path.join(OUTPUT_PATH, 'embedding_matrix.npy')

'''
python main.py --model_type LSTM --hidden_size 16 --dropout 0.8 --epoch 30
python main.py --model_type GRU --hidden_size 32 --dropout 0.9 --epoch 30
python main.py --model_type CNN --dropout 0.5 --epoch 10
'''