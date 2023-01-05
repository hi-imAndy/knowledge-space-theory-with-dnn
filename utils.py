
def get_models_path():
    return '../models/'


def log_to_file(text):
    log_file = open("../log.txt", "a")
    log_file.write(str(text) + "\n")
    log_file.close()
