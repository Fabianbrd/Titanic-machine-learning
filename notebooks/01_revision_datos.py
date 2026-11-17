import pandas as pd

# Ruta de los archivos
ruta = "data/raw/"

# Cargar los archivos
train = pd.read_csv(ruta + "train.csv")
test = pd.read_csv(ruta + "test.csv")
gender = pd.read_csv(ruta + "gender_submission.csv")

# Mostrar dimensiones
print("\nINFORMACION DE LOS ARCHIVOS")
print("---------------------------")

print("Train:", train.shape)
print("Test:", test.shape)
print("Gender Submission:", gender.shape)

# Mostrar primeros registros
print("\nPRIMEROS REGISTROS DEL TRAIN")
print(train.head())

# Comprobar valores nulos
print("\nVALORES NULOS")
print(train.isnull().sum())

# Comprobar duplicados
print("\nREGISTROS DUPLICADOS")
print(train.duplicated().sum())

print("\nREVISION COMPLETADA")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
train = pd.read_csv("data/raw/train.csv")

# Configuración visual
sns.set_theme(style="whitegrid")

# =====================================
# 1. INFORMACION GENERAL
# =====================================

print("\nINFORMACION GENERAL DEL DATASET")

print(train.info())

# =====================================
# 2. ESTADISTICAS DESCRIPTIVAS
# =====================================

print("\nESTADISTICAS DESCRIPTIVAS")

print(train.describe())

# =====================================
# 3. PORCENTAJE DE SUPERVIVENCIA
# =====================================

supervivencia = train["Survived"].value_counts(normalize=True) * 100

print("\nPORCENTAJE DE SUPERVIVENCIA")

print(supervivencia)

# =====================================
# 4. SUPERVIVENCIA POR SEXO
# =====================================

print("\nSUPERVIVENCIA POR SEXO")

print(train.groupby("Sex")["Survived"].mean() * 100)

# =====================================
# 5. SUPERVIVENCIA POR CLASE
# =====================================

print("\nSUPERVIVENCIA POR CLASE")

print(train.groupby("Pclass")["Survived"].mean() * 100)

# =====================================
# 6. GRAFICO DE SUPERVIVENCIA
# =====================================

plt.figure(figsize=(8, 5))

sns.countplot(data=train, x="Survived", hue="Survived")

plt.title("Distribucion de supervivencia")
plt.xlabel("Supervivencia (0 = No, 1 = Si)")
plt.ylabel("Numero de pasajeros")

plt.tight_layout()

plt.savefig("images/supervivencia.png")

plt.show()

# =====================================
# 7. SUPERVIVENCIA POR SEXO
# =====================================

plt.figure(figsize=(8, 5))

sns.barplot(data=train, x="Sex", y="Survived")

plt.title("Supervivencia segun el sexo")
plt.xlabel("Sexo")
plt.ylabel("Tasa de supervivencia")

plt.tight_layout()

plt.savefig("images/supervivencia_sexo.png")

plt.show()

print("\nANALISIS EXPLORATORIO COMPLETADO")
import pandas as pd
from pathlib import Path

# ==========================================
# PROYECTO TITANIC
# PREPARACION Y TRANSFORMACION DE DATOS
# ==========================================

# Rutas del proyecto
ruta = Path("data/raw")

# Cargar archivos originales
train = pd.read_csv(ruta / "train.csv")
test = pd.read_csv(ruta / "test.csv")

# ==========================================
# 1. CREAR NUEVAS VARIABLES
# ==========================================


def preparar_datos(df):

    df = df.copy()

    # Extraer titulo del nombre
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

    # Agrupar titulos poco frecuentes
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    titulos_frecuentes = ["Mr", "Miss", "Mrs", "Master"]

    df["Title"] = df["Title"].where(df["Title"].isin(titulos_frecuentes), "Other")

    # Crear tamaño familiar
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Identificar pasajeros que viajaban solos
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Identificar si disponian de cabina registrada
    df["HasCabin"] = (df["Cabin"].notna()).astype(int)

    return df


# Aplicar transformaciones
train = preparar_datos(train)
test = preparar_datos(test)

# ==========================================
# 2. ELIMINAR COLUMNAS NO NECESARIAS
# ==========================================

columnas_eliminar = ["Name", "Ticket", "Cabin", "PassengerId"]

train = train.drop(columns=columnas_eliminar)

test = test.drop(columns=columnas_eliminar)

# ==========================================
# 3. COMPROBAR RESULTADOS
# ==========================================

print("\n========== TRAIN PREPARADO ==========")

print(train.head())

print("\n========== DIMENSIONES ==========")

print("Train:", train.shape)
print("Test:", test.shape)

print("\n========== VARIABLES NUEVAS ==========")

print(train[["Title", "FamilySize", "IsAlone", "HasCabin"]].head())

print("\n========== VALORES AUSENTES ==========")

print(train.isnull().sum())

# ==========================================
# 4. GUARDAR DATOS PROCESADOS
# ==========================================

carpeta_salida = Path("data/processed")

carpeta_salida.mkdir(parents=True, exist_ok=True)

train.to_csv(carpeta_salida / "train_processed.csv", index=False)

test.to_csv(carpeta_salida / "test_processed.csv", index=False)

print("\nARCHIVOS PROCESADOS GUARDADOS")

print("\nPREPARACION COMPLETADA")
