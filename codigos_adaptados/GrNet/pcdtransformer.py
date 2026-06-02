import h5py
import open3d as o3d
import numpy as np
import os
import glob

# Rutas
input_folder = r'C:\Models\Completion\GRNet-master\output\benchmark\sorghum'
output_folder = r'C:\Models\Completion\GRNet-master\output\benchmark\sorghum_pcd'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Buscar todos los archivos .h5 generados
h5_files = glob.glob(os.path.join(input_folder, "*.h5"))

print(f"Encontrados {len(h5_files)} archivos para convertir...")

for h5_path in h5_files:
    file_name = os.path.basename(h5_path).replace('.h5', '.pcd')
    output_path = os.path.join(output_folder, file_name)
    
    with h5py.File(h5_path, 'r') as f:
        # El loader de GRNet guarda los datos bajo la clave 'data'
        # o directamente como el array principal dependiendo de la versión de IO.py
        if 'data' in f:
            data = f['data'][:]
        else:
            # Si no hay clave 'data', intentamos leer el primer dataset disponible
            key = list(f.keys())[0]
            data = f[key][:]
            
    # Crear objeto de nube de puntos de Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    
    # Guardar en formato .pcd
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Convertido: {file_name}")

print(f"\nProceso finalizado. Archivos guardados en: {output_folder}")