import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("Loading Forest Cover Type dataset...")
    covtype = fetch_covtype(as_frame=True)

    X: pd.DataFrame = covtype.data
    y: pd.Series = covtype.target

    print()
    print("Shape of X (features):", X.shape)
    print("Shape of y (target):", y.shape)
    print("First 5 rows of features:")
    print(X.head())
    print()

    df = X.copy()
    df["Cover_Type"] = y

    cover_type_names = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz",
    }

    print("Cover type codes and names:")
    for code, name in cover_type_names.items():
        print(f"  {code}: {name}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    importances = model.feature_importances_
    feature_importances = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("Top 15 features by importance:")
    print(feature_importances.head(15))
    print()

    print("Hypothesis 1: Higher elevation for Spruce/Fir compared to other types")
    elevation_means = (
        df.groupby("Cover_Type")["Elevation"]
        .mean()
        .to_frame(name="mean_elevation")
        .sort_values("mean_elevation", ascending=False)
    )
    print(elevation_means)
    print()

    print("Hypothesis 2: Soil type one hot columns among top important features")
    soil_features = [f for f in X.columns if "Soil_Type" in f]
    top_features = feature_importances.head(20)
    top_soil = top_features[top_features["feature"].isin(soil_features)]
    print("Soil type features that appear in the top 20 important features:")
    print(top_soil if not top_soil.empty else "None of the top 20 are soil features")
    print()

    print("Hypothesis 3: Distance to hydrology for Cottonwood/Willow vs others")
    cols_hydro = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
    ]

    if all(col in df.columns for col in cols_hydro):
        mask_cw = df["Cover_Type"] == 4
        mask_other = df["Cover_Type"] != 4

        means_cw = df.loc[mask_cw, cols_hydro].mean()
        means_other = df.loc[mask_other, cols_hydro].mean()

        print("Average distances to hydrology for Cottonwood/Willow (cover type 4):")
        print(means_cw)
        print()
        print("Average distances to hydrology for all other cover types:")
        print(means_other)
        print()
    else:
        print("Hydrology distance columns not found in this version of the dataset.")
        print()

    print("Done. Use these outputs to support the three hypotheses in your write up.")


if __name__ == "__main__":
    main()
