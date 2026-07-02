import open3d as o3d
import os

# Caminho do arquivo de teste
arquivo = "log_volta_da_ufes_20230522.txt_lidar/1684870000/1684870400/1684870436.981496.pointcloud"

if os.path.exists(arquivo):
    try:
        # Tenta ler o arquivo. Se ele não for um formato padrão (como PLY/PCD),
        # talvez precisemos especificar o formato.
        pcd = o3d.io.read_point_cloud(arquivo)
        
        if pcd.is_empty():
            print("O Open3D leu o arquivo, mas ele parece vazio ou formato desconhecido.")
        else:
            print(f"Sucesso! Pontos carregados: {len(pcd.points)}")
            # Exibe os primeiros 5 pontos
            print(f"Primeiros pontos: {np.asarray(pcd.points)[:5]}")
            
    except Exception as e:
        print(f"Erro ao ler com Open3D: {e}")
else:
    print("Arquivo de teste não encontrado.")
