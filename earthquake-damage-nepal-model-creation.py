import sqlite3
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)




def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
        SELECT distinct(i.building_id) AS b_id,
           s.*,
           d.damage_grade
        FROM id_map AS i
        JOIN building_structure AS s ON i.building_id = s.building_id
        JOIN building_damage AS d ON i.building_id = d.building_id
        WHERE district_id = 4
    """

    # Read query results into DataFrame
    df = pd.read_sql(query, conn, index_col="b_id")

    # identify leaky columns
    drop_cols = [col for col in df.columns if "post_eq" in col]

    # create binary target
    df["damage_grade"] = df["damage_grade"].str[-1].astype(int)
    df["severe_damage"] = (df["damage_grade"] > 3).astype(int)

    # drop old target
    drop_cols.append("damage_grade")

    # drop multicollenearity column
    drop_cols.append("count_floors_pre_eq")

    # drop high cardinality
    drop_cols.append("building_id")

    # drop columns
    df.drop(columns=drop_cols, inplace=True)

    return df





def model_():
    df = wrangle("nepal.sqlite")
    target = "severe_damage"
    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        LogisticRegression(max_iter=5000)
    )

    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = model.score(X_test, y_test)

    print("Training Accuracy:", round(acc_train, 2))
    print("Test Accuracy:", round(acc_test, 2))

    # Fit model to training data
    with open("model-nepal-damage-prediction.pkl", "wb") as f:
        pickle.dump(model, f)


    X_test.to_csv("X_test_nepal-damage-prediction.csv", index=False)

    return print("Model Created")

model_()