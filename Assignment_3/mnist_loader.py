from pathlib import Path
import idx2numpy

BASE_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"

def load_mnist():
    train_images = idx2numpy.convert_from_file(str(DATA_DIRECTORY/ "train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(str(DATA_DIRECTORY / "train-labels-idx1-ubyte"))

    test_images = idx2numpy.convert_from_file(str(DATA_DIRECTORY/ "t10k-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(str(DATA_DIRECTORY/"t10k-labels-idx1-ubyte"))

    return train_images, train_labels, test_images, test_labels

