import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ==========================================
# 1. CONFIGURACION GENERAL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ruta_datos = BASE_DIR / "data" / "raw" / "train.csv"

carpeta_imagenes = BASE_DIR / "images"
carpeta_imagenes.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(ruta_datos)

sns.set_theme(style="whitegrid", context="talk", font="sans-serif")

AZUL = "#17365D"
TURQUESA = "#20A39E"
GRIS = "#7A8793"
CLARO = "#DCE6F1"

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.titleweight"] = "bold"


# ==========================================
# 2. ANALISIS GENERAL
# ==========================================

print("\nINFORMACION GENERAL")
train.info()

print("\nESTADISTICAS DESCRIPTIVAS")
print(train.describe())

print("\nPORCENTAJE DE SUPERVIVENCIA")
print(train["Survived"].value_counts(normalize=True) * 100)

print("\nSUPERVIVENCIA POR SEXO")
print(train.groupby("Sex")["Survived"].mean() * 100)

print("\nSUPERVIVENCIA POR CLASE")
print(train.groupby("Pclass")["Survived"].mean() * 100)


# ==========================================
# 3. GRAFICA GENERAL DE SUPERVIVENCIA
# ==========================================

conteo = train["Survived"].value_counts().sort_index()

etiquetas = ["No sobrevivieron", "Sobrevivieron"]
colores = [GRIS, TURQUESA]

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(etiquetas, conteo.values, color=colores, width=0.55)

total = len(train)

for barra, valor in zip(barras, conteo.values):
    porcentaje = valor / total * 100

    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 10,
        f"{valor} pasajeros\n({porcentaje:.1f}%)",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

ax.set_title("Distribucion general de supervivencia", pad=20)
ax.set_ylabel("Numero de pasajeros")
ax.set_xlabel("")
ax.set_ylim(0, max(conteo.values) * 1.2)

sns.despine()

plt.tight_layout()
plt.savefig(
    carpeta_imagenes / "01_supervivencia_general.png", dpi=300, bbox_inches="tight"
)
plt.close()


# ==========================================
# 4. SUPERVIVENCIA POR SEXO
# ==========================================

sexo = train.groupby("Sex")["Survived"].mean().mul(100).reindex(["female", "male"])

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(["Mujeres", "Hombres"], sexo.values, color=[TURQUESA, AZUL], width=0.55)

for barra, valor in zip(barras, sexo.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 2,
        f"{valor:.1f}%",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

ax.set_title("Tasa de supervivencia por sexo", pad=20)
ax.set_ylabel("Supervivencia (%)")
ax.set_ylim(0, 110)

sns.despine()

plt.tight_layout()
plt.savefig(
    carpeta_imagenes / "02_supervivencia_sexo.png", dpi=300, bbox_inches="tight"
)
plt.close()


# ==========================================
# 5. SUPERVIVENCIA POR CLASE
# ==========================================

clase = train.groupby("Pclass")["Survived"].mean().mul(100)

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    ["Primera clase", "Segunda clase", "Tercera clase"],
    clase.values,
    color=[TURQUESA, AZUL, GRIS],
    width=0.55,
)

for barra, valor in zip(barras, clase.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 2,
        f"{valor:.1f}%",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

ax.set_title("Supervivencia según clase del pasajero", pad=20)
ax.set_ylabel("Supervivencia (%)")
ax.set_ylim(0, 100)

sns.despine()

plt.tight_layout()
plt.savefig(
    carpeta_imagenes / "03_supervivencia_clase.png", dpi=300, bbox_inches="tight"
)
plt.close()


# ==========================================
# 6. SUPERVIVENCIA POR GRUPOS DE EDAD
# ==========================================

train["GrupoEdad"] = pd.cut(
    train["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["0-12 años", "13-18 años", "19-35 años", "36-60 años", "Más de 60"],
    include_lowest=True,
)

edad = train.groupby("GrupoEdad", observed=False)["Survived"].mean().mul(100).dropna()

fig, ax = plt.subplots(figsize=(11, 6))

barras = ax.bar(edad.index.astype(str), edad.values, color=TURQUESA, width=0.6)

for barra, valor in zip(barras, edad.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 2,
        f"{valor:.1f}%",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

ax.set_title("Supervivencia por grupos de edad", pad=20)
ax.set_ylabel("Supervivencia (%)")
ax.set_xlabel("Grupo de edad")
ax.set_ylim(0, 100)

sns.despine()

plt.tight_layout()
plt.savefig(
    carpeta_imagenes / "04_supervivencia_edad.png", dpi=300, bbox_inches="tight"
)
plt.close()


# ==========================================
# 7. DISTRIBUCION DE EDADES
# ==========================================

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    data=train, x="Age", bins=30, kde=True, color=AZUL, edgecolor="white", ax=ax
)

ax.set_title("Distribucion de edades de los pasajeros", pad=20)
ax.set_xlabel("Edad")
ax.set_ylabel("Numero de pasajeros")

sns.despine()

plt.tight_layout()
plt.savefig(
    carpeta_imagenes / "05_distribucion_edades.png", dpi=300, bbox_inches="tight"
)
plt.close()


# ==========================================
# 8. RESULTADO FINAL
# ==========================================

print("\n====================================")
print("ANALISIS EXPLORATORIO COMPLETADO")
print("====================================")

print("\nGRAFICOS GENERADOS:")

for imagen in sorted(carpeta_imagenes.glob("*.png")):
    print("-", imagen.name)

print("\nTodos los graficos se han guardado correctamente.")
