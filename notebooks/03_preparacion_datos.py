import pandas as pd
from pathlib import Path

# Cargar archivos originales
ruta = Path("data/raw")

train = pd.read_csv(ruta / "train.csv")
test = pd.read_csv(ruta / "test.csv")


# Preparar variables
def preparar_datos(df):

    df = df.copy()

    # Extraer el título del nombre
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    titulos_frecuentes = ["Mr", "Miss", "Mrs", "Master"]

    df["Title"] = df["Title"].where(df["Title"].isin(titulos_frecuentes), "Other")

    # Crear variables familiares
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Identificar si el pasajero tenía cabina registrada
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    return df


# Aplicar preparación
train = preparar_datos(train)
test = preparar_datos(test)


# Eliminar columnas que no utilizaremos directamente
columnas_eliminar = ["Name", "Ticket", "Cabin", "PassengerId"]

train = train.drop(columns=columnas_eliminar)
test = test.drop(columns=columnas_eliminar)


# Mostrar resultados
print("\n========== TRAIN PREPARADO ==========")
print(train.head())

print("\n========== DIMENSIONES ==========")
print("Train:", train.shape)
print("Test:", test.shape)

print("\n========== VARIABLES NUEVAS ==========")
print(train[["Title", "FamilySize", "IsAlone", "HasCabin"]].head())

print("\n========== VALORES AUSENTES ==========")
print(train.isnull().sum())


# Guardar archivos preparados
carpeta_salida = Path("data/processed")
carpeta_salida.mkdir(parents=True, exist_ok=True)

train.to_csv(carpeta_salida / "train_processed.csv", index=False)
test.to_csv(carpeta_salida / "test_processed.csv", index=False)

print("\nARCHIVOS PROCESADOS GUARDADOS")
print("\nPREPARACION COMPLETADA")
