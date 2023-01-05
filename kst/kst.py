import pandas as pd
from learning_spaces.kst import iita


def create_ks(answers: dict):
    data_frame = pd.DataFrame(answers)
    response = iita(data_frame, v=1)
    return response
