import os
import json
import glob

BASE_PATH = r'C:\Users\Joanm\Downloads\PlantPCom\PlantPCom'
TAXONOMY_ID = 'sorghum'
JSON_PATH = os.path.join('datasets', 'Sorghum.json')

def procesar():
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    datos_json = {
        "taxonomy_id": TAXONOMY_ID,
        "taxonomy_name": TAXONOMY_ID,
        "train": [], "val": [], "test": []
    }

    subsets = ['train', 'val', 'test']

    for ss in subsets:
        path_complete = os.path.join(BASE_PATH, ss, 'complete', TAXONOMY_ID)
        path_partial = os.path.join(BASE_PATH, ss, 'partial', TAXONOMY_ID)

        if not os.path.exists(path_complete):
            print(f"--- Saltando {ss}: No existe la ruta {path_complete}")
            continue

        # Obtener todos los modelos .pcd de la carpeta complete
        modelos = [os.path.splitext(f)[0] for f in os.listdir(path_complete) if f.endswith('.pcd')]
        
        for m_id in modelos:
            folder_parcial = os.path.join(path_partial, m_id)
            
            if os.path.exists(folder_parcial):
                # RENOMBRAR VISTAS A 00, 01, 02... (Lo que el modelo exige)
                vistas = sorted([f for f in os.listdir(folder_parcial) if f.endswith('.pcd')])
                for idx, nombre_viejo in enumerate(vistas):
                    viejo_path = os.path.join(folder_parcial, nombre_viejo)
                    nuevo_path = os.path.join(folder_parcial, f"{idx:02d}.pcd")
                    if viejo_path != nuevo_path:
                        os.rename(viejo_path, nuevo_path)
                
                # Agregar al JSON
                datos_json[ss].append(m_id)
                print(f"Procesado modelo: {m_id} en {ss}")
            else:
                print(f"!!! Error: No hay carpeta parcial para {m_id} en {ss}")

    # Guardar el JSON (Formato: Lista de diccionarios)
    with open(JSON_PATH, 'w') as f:
        json.dump([datos_json], f, indent=4)
    
    print(f"\nPROCESO TERMINADO.")
    print(f"JSON generado en: {JSON_PATH}")
    print(f"Resumen: Train({len(datos_json['train'])}), Val({len(datos_json['val'])}), Test({len(datos_json['test'])})")

if __name__ == "__main__":
    procesar()