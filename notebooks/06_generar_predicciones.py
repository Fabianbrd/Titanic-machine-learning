import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. CARGAR DATOS
# ==========================================

train = pd.read_csv("data/processed/train_processed.csv")

# Recuperar los identificadores originales
test_original = pd.read_csv("data/raw/test.csv")

test = pd.read_csv("data/processed/test_processed.csv")


# ==========================================
# 2. DEFINIR VARIABLES
# ==========================================

X_train = train.drop(columns=["Survived"])
y_train = train["Survived"]

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
# 3. PREPARAR DATOS
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
# 4. ENTRENAR MODELO FINAL
# ==========================================

modelo = Pipeline(
    steps=[
        ("preparacion", preprocesador),
        ("modelo", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

modelo.fit(X_train, y_train)


# ==========================================
# 5. GENERAR PREDICCIONES
# ==========================================

predicciones = modelo.predict(test)


# ==========================================
# 6. CREAR ARCHIVO KAGGLE
# ==========================================

submission = pd.DataFrame(
    {"PassengerId": test_original["PassengerId"], "Survived": predicciones.astype(int)}
)


# ==========================================
# 7. GUARDAR RESULTADO
# ==========================================

carpeta = Path("outputs")
carpeta.mkdir(parents=True, exist_ok=True)

submission.to_csv(carpeta / "submission.csv", index=False)


# ==========================================
# 8. COMPROBAR RESULTADOS
# ==========================================

print("\n===================================")
print("PREDICCIONES GENERADAS")
print("===================================")

print("\nPRIMEROS REGISTROS:")
print(submission.head(10))

print("\nDIMENSIONES:")
print(submission.shape)

print("\nDISTRIBUCION DE PREDICCIONES:")
print(submission["Survived"].value_counts())

print("\nVALORES NULOS:")
print(submission.isnull().sum())

print("\nARCHIVO GUARDADO EN: outputs/submission.csv")

print("\nPROCESO COMPLETADO")
