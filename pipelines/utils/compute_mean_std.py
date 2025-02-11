import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pipelines.utils import compute_mean_std
import yaml

# Cargar parámetros desde params.yaml
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    data_path = params["data"]["image_folder_path"]  # Ruta de imágenes desde params.yaml
    mean_std_output = sys.argv[1]  # Ruta del archivo mean_std.csv desde argumento del sistema

    # Calcular mean y std
    mean, std = compute_mean_std(data_path)

    # Guardar en CSV
    pd.DataFrame([mean, std]).to_csv(mean_std_output, index=False, header=False)
