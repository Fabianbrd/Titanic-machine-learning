import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
# 4. PREPROCESAMIENTO
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
# 5. CREAR PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preparacion", preprocesador),
        ("modelo", RandomForestClassifier(random_state=42)),
    ]
)


# ==========================================
# 6. CONFIGURAR PARAMETROS
# ==========================================

parametros = {
    "modelo__n_estimators": [100, 200],
    "modelo__max_depth": [5, 10, None],
    "modelo__min_samples_split": [2, 5],
    "modelo__min_samples_leaf": [1, 2],
}


# ==========================================
# 7. BUSQUEDA DE MEJORES PARAMETROS
# ==========================================

validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

busqueda = GridSearchCV(
    estimator=pipeline,
    param_grid=parametros,
    scoring="f1",
    cv=validacion,
    n_jobs=1,
    verbose=1,
)

busqueda.fit(X_train, y_train)


# ==========================================
# 8. EVALUAR MEJOR MODELO
# ==========================================

mejor_modelo = busqueda.best_estimator_

predicciones = mejor_modelo.predict(X_valid)

print("\n===================================")
print("OPTIMIZACION RANDOM FOREST")
print("===================================")

print("\nMEJORES PARAMETROS:")
print(busqueda.best_params_)

print("\nRESULTADOS DE VALIDACION:")

print("Accuracy:", round(accuracy_score(y_valid, predicciones), 4))

print("Precision:", round(precision_score(y_valid, predicciones), 4))

print("Recall:", round(recall_score(y_valid, predicciones), 4))

print("F1-Score:", round(f1_score(y_valid, predicciones), 4))

print("\nOPTIMIZACION COMPLETADA")
