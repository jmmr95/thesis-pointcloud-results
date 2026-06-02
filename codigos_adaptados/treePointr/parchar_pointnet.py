import os
import glob

# 1. Descargar el repositorio original
if not os.path.exists("Pointnet2_PyTorch"):
    os.system("git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git")

# 2. Localizar todos los archivos de C++ y CUDA
archivos = glob.glob("Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/*") + \
           glob.glob("Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/include/*")

# 3. Aplicar el parche de compatibilidad para PyTorch 2.4
for f in archivos:
    if f.endswith('.cpp') or f.endswith('.cu') or f.endswith('.h'):
        with open(f, 'r', encoding='utf-8') as file:
            contenido = file.read()
        
        # Reemplazar la macro obsoleta por la actual
        contenido = contenido.replace('AT_CHECK', 'TORCH_CHECK')
        
        with open(f, 'w', encoding='utf-8') as file:
            file.write(contenido)

print("El código está listo para compilar.")