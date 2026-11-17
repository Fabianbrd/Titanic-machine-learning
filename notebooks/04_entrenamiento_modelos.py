import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================================
# 1. CARGAR DATOS
# ==========================================

train = pd.read_csv("data/processed/train_processed.csv")

X = train.drop(columns=["Survived"])
y = train["Survived"]


# ==========================================
# 2. DIVIDIR DATOS
# ==========================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ==========================================
# 3. DEFINIR VARIABLES
# ==========================================

columnas_numericas = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
]

columnas_categoricas = ["Pclass", "Sex", "Embarked", "Title"]


# ==========================================
# 4. PREPARAR DATOS
# ==========================================

procesador_numerico = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ]
)

procesador_categorico = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocesador = ColumnTransformer(
    transformers=[
        ("numerico", procesador_numerico, columnas_numericas),
        ("categorico", procesador_categorico, columnas_categoricas),
    ]
)


# ==========================================
# 5. CREAR MODELOS
# ==========================================

modelos = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}


# ==========================================
# 6. ENTRENAR Y EVALUAR
# ==========================================

resultados = []

for nombre, algoritmo in modelos.items():

    modelo = Pipeline(steps=[("preparacion", preprocesador), ("modelo", algoritmo)])

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_valid)

    accuracy = accuracy_score(y_valid, predicciones)
    precision = precision_score(y_valid, predicciones)
    recall = recall_score(y_valid, predicciones)
    f1 = f1_score(y_valid, predicciones)

    resultados.append(
        {
            "Modelo": nombre,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
    )

    print("\n===================================")
    print(nombre)
    print("===================================")

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    print("\nMATRIZ DE CONFUSION")
    print(confusion_matrix(y_valid, predicciones))

    print("\nINFORME DE CLASIFICACION")
    print(classification_report(y_valid, predicciones))


# ==========================================
# 7. COMPARAR RESULTADOS
# ==========================================

tabla_resultados = pd.DataFrame(resultados)

print("\n===================================")
print("COMPARACION FINAL DE MODELOS")
print("===================================")

print(
    tabla_resultados.sort_values(by="F1-Score", ascending=False).to_string(index=False)
)

print("\nENTRENAMIENTO COMPLETADO")
