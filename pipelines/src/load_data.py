import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pipelines.utils.data_utils import load_data_image_folder
from pipelines.utils import transform_tensor
import yaml
import pandas as pd
import torch

# Cargar parámetros desde params.yaml
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    processed_path = params["data"]["processed"]  # Ruta de datos procesados desde params.yaml
    data_path = params["data"]["image_folder_path"]  # Ruta de imágenes desde params.yaml
    mean_std_filename = sys.argv[1]  # Nombre del archivo mean_std.csv desde argumento del sistema
    output_file = sys.argv[2]  # Ruta para guardar los datos procesados

    # Construir ruta completa para mean_std.csv
    mean_std_path = os.path.join(processed_path, mean_std_filename)

    num_workers = os.cpu_count()

    # Leer mean y std desde el archivo CSV
    mean, std = pd.read_csv(mean_std_path, header=None).values
    mean, std = torch.tensor(mean), torch.tensor(std)

    # Aplicar transformación y cargar datos
    default_transform = transform_tensor(mean, std)
    loaded_data = load_data_image_folder(data_path, default_transform, num_workers)

    # Guardar datos procesados
    torch.save(loaded_data, os.path.join(processed_path, output_file))

