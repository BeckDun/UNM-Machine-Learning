from pathlib import Path
import idx2numpy
import matplotlib.pyplot as plt

BASE_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"

def load_fahion_mnist():
    train_images = idx2numpy.convert_from_file(str(DATA_DIRECTORY/ "fashion_train_images"))
    train_labels = idx2numpy.convert_from_file(str(DATA_DIRECTORY/ "fashion_train_labels"))
    
    test_images = idx2numpy.convert_from_file(str(DATA_DIRECTORY/ "fashion_test_images"))
    test_labels = idx2numpy.convert_from_file(str(DATA_DIRECTORY/ "fashion_test_labels"))

    return train_images, train_labels, test_images, test_labels

