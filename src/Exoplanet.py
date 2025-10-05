import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from ipywidgets import interact, FloatSlider, Label

SEED = 42

url = 'https://raw.githubusercontent.com/Siddharths99/NASA-Exoplanet-Detection/refs/heads/main/data/kepler.csv'
df = pd.read_csv(url)
df_names = df[["kepler_name","kepoi_name"]].reset_index(drop=True)
print("Initial dataset size:", df.shape)
df.head(3)

drop_cols = ["rowid","kepid","kepoi_name","kepler_name",
             "koi_pdisposition","koi_tce_delivname","ra","dec"]
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

df_model = df_model.dropna(subset=["koi_disposition"])
df_model["koi_disposition"] = df_model["koi_disposition"].str.upper().str.strip()

num_cols = df_model.select_dtypes(include=np.number).columns.tolist()
df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].median())

print(df_model["koi_disposition"].value_counts())
print("Dataset size after cleaning:", len(df_model))

feature_cols = ["koi_period","koi_duration","koi_prad","koi_teq","koi_insol",
                "koi_impact","koi_depth","koi_steff","koi_slogg","koi_srad","koi_model_snr"]
feature_cols = [c for c in feature_cols if c in df_model.columns]

X = df_model[feature_cols]
y = df_model["koi_disposition"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)

smote = SMOTE(random_state=SEED)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Train size after SMOTE:", X_train.shape)
print("Test size:", X_test.shape)

clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

print(classification_report(y_test, y_pred, target_names=le.classes_))

df_test = X_test.copy()
df_test["Actual"] = le.inverse_transform(y_test)
df_test["Predicted"] = le.inverse_transform(y_pred)
df_test["Probability"] = clf.predict_proba(X_test).max(axis=1)
df_test = pd.concat([df_names.iloc[df_test.index].reset_index(drop=True), df_test.reset_index(drop=True)], axis=1)

df_test.to_csv("exo_predictions_with_names.csv", index=False)
top_candidates = df_test[df_test["Predicted"].isin(["CONFIRMED","CANDIDATE"])].sort_values(by="Probability", ascending=False)
top_candidates.to_csv("top_exo_candidates_with_names.csv", index=False)

# Top 10 candidates table
top10 = top_candidates.head(10)
display(HTML("<h3 style='color:#4CAF50'>Top 10 Candidate/Confirmed Exoplanets</h3>"))
display(top10[['kepler_name','Predicted','Probability','koi_prad','koi_period','koi_teq']])

# Feature importance
plt.figure(figsize=(8,6))
feat_importances.sort_values().plot(kind='barh', color="#2196F3")
plt.title("Feature Importances")
plt.show()

# Confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.show()

# Top candidate bar chart
plt.figure(figsize=(10,6))
sns.barplot(x="Probability", y=top10.index, hue="Predicted", data=top10, dodge=False, palette=["#4CAF50","#FFC107"])
plt.title("Top 10 Candidate/Confirmed Exoplanets")
plt.show()

# Probability heatmap
proba_df = pd.DataFrame(clf.predict_proba(X_test), columns=le.classes_)
plt.figure(figsize=(12,6))
sns.heatmap(proba_df.head(15).T, annot=True, cmap="YlGnBu")
plt.title("Prediction Probabilities for First 15 Test Samples")
plt.show()

# Classification report and accuracy
print(classification_report(y_test, clf.predict(X_test), target_names=le.classes_))
print("Accuracy:", clf.score(X_test, y_test))

def predict_new_exoplanet(features_dict):
    df_new = pd.DataFrame([features_dict])
    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0
    df_new = df_new[feature_cols]
    pred_class = le.inverse_transform(clf.predict(df_new))[0]
    pred_prob = clf.predict_proba(df_new).max()
    return pred_class, pred_prob

def interactive_predict(koi_period, koi_duration, koi_prad, koi_teq, koi_insol,
                        koi_impact, koi_depth, koi_steff, koi_slogg, koi_srad, koi_model_snr):
    features_dict = {
        "koi_period": koi_period,
        "koi_duration": koi_duration,
        "koi_prad": koi_prad,
        "koi_teq": koi_teq,
        "koi_insol": koi_insol,
        "koi_impact": koi_impact,
        "koi_depth": koi_depth,
        "koi_steff": koi_steff,
        "koi_slogg": koi_slogg,
        "koi_srad": koi_srad,
        "koi_model_snr": koi_model_snr
    }
    pred_class, pred_prob = predict_new_exoplanet(features_dict)
    display(Label(f"Predicted Class: {pred_class}"))
    display(Label(f"Prediction Probability: {pred_prob:.2f}"))

interact(interactive_predict,
         koi_period=FloatSlider(value=365, min=0, max=1000, step=0.1),
         koi_duration=FloatSlider(value=10, min=0, max=50, step=0.1),
         koi_prad=FloatSlider(value=1, min=0, max=20, step=0.1),
         koi_teq=FloatSlider(value=300, min=50, max=5000, step=10),
         koi_insol=FloatSlider(value=1, min=0, max=1000, step=0.1),
         koi_impact=FloatSlider(value=0, min=0, max=5, step=0.01),
         koi_depth=FloatSlider(value=0.01, min=0, max=5, step=0.01),
         koi_steff=FloatSlider(value=5800, min=2000, max=10000, step=10),
         koi_slogg=FloatSlider(value=4.4, min=0, max=6, step=0.01),
         koi_srad=FloatSlider(value=1.0, min=0, max=10, step=0.01),
         koi_model_snr=FloatSlider(value=10, min=0, max=100, step=1)
)