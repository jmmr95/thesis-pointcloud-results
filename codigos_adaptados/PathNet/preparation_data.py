import os
import numpy as np
import open3d as o3d
import h5py
from pathlib import Path

def normalize_pc(points):
    """Centra la nube y la escala al rango [-1, 1]."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furtherst_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if furtherst_distance > 0:
        points /= furtherst_distance
    return points

def add_gaussian_noise(points, sigma=0.01):
    """Añade ruido gaussiano para generar el dataset de entrada."""
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise

def preparar_dataset_pathnet():
    # 1. Configuración de Rutas
    # Ruta de origen de archivos .ply
    source_path = Path(r"C:\Users\Joanm\Downloads\Crops 3D\Crops3D_10k\Maize")
    
    # Carpeta 'data' dentro del proyecto (donde está el código de PathNet)
    project_root = Path(__file__).parent.absolute()
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "train _data.hdf5"

    # 2. Selección de Archivos (40 train + 10 val + 10 test = 60 archivos)
    all_files = sorted([f for f in os.listdir(source_path) if f.endswith('.ply')])
    if len(all_files) < 60:
        print(f"Error: Se necesitan 60 archivos, pero solo hay {len(all_files)}.")
        return
    
    selected_files = all_files[:60]

    inputs_batch = [] # Nubes ruidosas
    target_batch = [] # Nubes limpias
    label_batch  = [] # Array de ceros (clase)

    print(f"Iniciando empaquetado en: {output_file}")

    for i, filename in enumerate(selected_files):
        # Cargar nube de puntos
        pcd = o3d.io.read_point_cloud(str(source_path / filename))
        points = np.asarray(pcd.points)

        # Asegurar exactamente 10,000 puntos
        if len(points) > 10000:
            points = points[:10000]
        elif len(points) < 10000:
            idx = np.random.choice(len(points), 10000 - len(points))
            points = np.vstack([points, points[idx]])

        # --- PROCESO ---
        # a. Normalización
        clean_points = normalize_pc(points)
        
        # b. Ruido (Será el 'input')
        noisy_points = add_gaussian_noise(clean_points, sigma=0.01)

        target_batch.append(clean_points)
        inputs_batch.append(noisy_points)
        label_batch.append(0.0) 
        
        print(f"[{i+1}/60] Procesado: {filename}", end='\r')

    # 3. Guardar en formato HDF5 compatible con PathNet
    with h5py.File(output_file, 'w') as f:
        # El DataLoader.py busca exactamente estos nombres:
        f.create_dataset('inputs', data=np.array(inputs_batch).astype(np.float32), compression='gzip')
        f.create_dataset('target', data=np.array(target_batch).astype(np.float32), compression='gzip')
        f.create_dataset('label',  data=np.array(label_batch).astype(np.float32), compression='gzip')

    print(f"\n\n¡PROCESO FINALIZADO!")
    print(f"Archivo generado: {output_file}")
    print(f"Contenido: 60 muestras con 'inputs', 'target' y 'label'.")

if __name__ == "__main__":
    preparar_dataset_pathnet()