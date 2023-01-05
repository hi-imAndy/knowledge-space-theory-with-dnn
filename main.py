from dnn.train import train_models
from kst.kst import create_ks
from utils import log_to_file

if __name__ == "__main__":
    answers = train_models()
    ks = create_ks(answers)
    log_to_file(ks)
