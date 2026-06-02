import os
import random
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

def convertir_ply_a_xyz(ruta_origen, ruta_destino):
    """Extrae coordenadas XYZ y las guarda en formato de texto .xyz."""
    try:
        plydata = PlyData.read(ruta_origen)
        vertices = np.stack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ], axis=1)
        # Formato de 8 decimales
        np.savetxt(ruta_destino, vertices, fmt='%.8f')
        return True
    except Exception as e:
        print(f"\nError en {ruta_origen}: {e}")
        return False

def preparar_todo():
    ruta_fuente = input("Introduce la ruta de la carpeta con los 225 archivos .ply: ").strip()
    if not os.path.isdir(ruta_fuente):
        print("Error: La ruta no es válida.")
        return

    # 1. Buscar todos los archivos .ply
    todos_los_archivos = []
    for root, dirs, files in os.walk(ruta_fuente):
        for f in files:
            if f.endswith('.ply'):
                todos_los_archivos.append(os.path.join(root, f))
    
    total = len(todos_los_archivos)
    print(f"Se encontraron {total} archivos.")
    random.shuffle(todos_los_archivos)

    # 2. Definir proporciones (70% Train, 15% Val, 15% Test)
    n_train = int(total * 0.70)  # ~157
    n_val = int(total * 0.15)    # ~34
    # El resto es para Test (~34)

    splits = {
        "train": todos_los_archivos[:n_train],
        "val": todos_los_archivos[n_train:n_train + n_val],
        "test": todos_los_archivos[n_train + n_val:]
    }

    # 3. Crear estructura de carpetas de IterativePFN
    res = "10000_poisson"
    # Carpeta de entrenamiento
    dir_train = os.path.join("data", "crops3d", "pointclouds", "train", res)
    # Carpeta de validación y Ground Truth de test
    dir_test_gt = os.path.join("data", "crops3d", "pointclouds", "test", res)
    # Carpeta de entrada para el script de inferencia (test.py)
    dir_test_input = os.path.join("data", "examples", f"crops3d_{res}_0.01")

    for d in [dir_train, dir_test_gt, dir_test_input]:
        os.makedirs(d, exist_ok=True)

    # 4. Procesar y mover
    print("\nProcesando y convirtiendo archivos...")
    
    for split, lista in splits.items():
        desc = f"Copiando a {split}"
        for ruta_orig in tqdm(lista, desc=desc):
            # Crear nombre único usando la carpeta madre (categoría) para evitar duplicados
            categoria = os.path.basename(os.path.dirname(ruta_orig))
            nombre_base = os.path.basename(ruta_orig).replace(".ply", ".xyz")
            nombre_final = f"{categoria}_{nombre_base}"

            if split == "train":
                dest = os.path.join(dir_train, nombre_final)
                convertir_ply_a_xyz(ruta_orig, dest)
            
            elif split == "val":
                # La validación durante el entrenamiento usa la carpeta 'test'
                dest = os.path.join(dir_test_gt, nombre_final)
                convertir_ply_a_xyz(ruta_orig, dest)

            elif split == "test":
                # Para el test final necesitamos el archivo en dos sitios:
                # 1. En 'examples' para que el modelo lo limpie
                dest_in = os.path.join(dir_test_input, nombre_final)
                convertir_ply_a_xyz(ruta_orig, dest_in)
                # 2. En 'pointclouds/test' para que el evaluador compare
                dest_gt = os.path.join(dir_test_gt, nombre_final)
                convertir_ply_a_xyz(ruta_orig, dest_gt)

    print(f"\nProceso finalizado:")
    print(f"- Entrenamiento: {len(splits['train'])} archivos en {dir_train}")
    print(f"- Validación: {len(splits['val'])} archivos en {dir_test_gt}")
    print(f"- Test: {len(splits['test'])} archivos listos para evaluar")

if __name__ == "__main__":
    preparar_todo()