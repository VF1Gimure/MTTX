import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pickle
import torch
from facenet_pytorch import MTCNN
from pipelines.utils.face_detection_utils import extract_mtcnn_boxes
import yaml
import pandas as pd

with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data.pt
    output_path = os.path.join(params["data"]["processed"], sys.argv[2])  # mtcnn_boxes.pkl
    mean_std_path = os.path.join(params["data"]["processed"], sys.argv[3])  # mean_std.csv

    num_workers = os.cpu_count()
    # Cargar datos
    loaded_data = torch.load(input_path)
    mean, std = pd.read_csv(mean_std_path, header=None).values

    # Convertir a tensores
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    # Inicializar MTCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(select_largest=False, post_process=False, device=device)

    # Extraer MTCNN bounding boxes
    filter_boxes = extract_mtcnn_boxes(mtcnn, loaded_data, mean, std,
                                       num_workers, 0)

    # Guardar las cajas detectadas
    with open(output_path, "wb") as f:
        pickle.dump(filter_boxes, f)

    print(f"MTCNN boxes guardados en: {output_path}")
