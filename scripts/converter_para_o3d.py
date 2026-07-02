import open3d as o3d
import numpy as np
import os

def carregar_binario_para_o3d(caminho):
    # Lê os dados como float32
    dados = np.fromfile(caminho, dtype=np.float32)
    
    # Redimensiona para (N, 4) conforme descobrimos (X, Y, Z, Intensidade)
    pontos_com_intensidade = dados.reshape(-1, 4)
    
    # Extrai apenas as 3 primeiras colunas (X, Y, Z) para a geometria
    pontos = pontos_com_intensidade[:, :3]
    
    # Cria o objeto de nuvem de pontos do Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pontos)
    return pcd

# Caminho do arquivo de teste
arquivo = "log_volta_da_ufes_20230522.txt_lidar/1684870000/1684870400/1684870436.981496.pointcloud"

if os.path.exists(arquivo):
    pcd = carregar_binario_para_o3d(arquivo)
    print(f"Sucesso! Nuvem criada com {len(pcd.points)} pontos.")
    
    # Para visualizar (remova o '#' da linha abaixo se estiver em ambiente gráfico):
    # o3d.visualization.draw_geometries([pcd])
else:
    print("Arquivo não encontrado.")
