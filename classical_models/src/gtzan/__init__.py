from gtzan.data.make_dataset import make_dataset_dl
from gtzan.data.make_dataset import make_dataset_ml
from gtzan.utils import majority_voting
from gtzan.utils import get_genres
from joblib import load
from tensorflow.keras.models import load_model

__all__ = ['AppManager']


class AppManager:
    def __init__(self, args, genres):
        self.args = args
        self.genres = genres

    def run(self):
        X = make_dataset_ml(self.args)
        pipe = load(self.args.model)
        tmp = pipe.predict(X)
        pred = get_genres(tmp[0], self.genres)
        print("predicted: {}".format(pred))
