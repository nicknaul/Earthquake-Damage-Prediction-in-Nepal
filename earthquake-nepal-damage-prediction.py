import warnings
import pandas as pd
warnings.simplefilter(action="ignore", category=FutureWarning)
import pickle




def make_prediction():
    df_nepal_predict = pd.read_csv("X_test_nepal-damage-prediction.csv", encoding='latin-1')
    with open("model-nepal-damage-prediction.pkl", "rb") as f:
        model = pickle.load(f)
    predict_damage = pd.Series(model.predict(df_nepal_predict))

    print(predict_damage.head(20))


make_prediction()

