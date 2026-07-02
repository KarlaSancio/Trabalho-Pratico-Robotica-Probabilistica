import pandas as pd
import numpy as np
import os
import glob

# Ajusta o caminho para buscar o arquivo CSV um nível acima (na raiz do projeto)
def carregar_trajetoria():
    # Procura o arquivo na raiz ou na pasta data
    caminho = "data/trajetoria_estimada.csv" if os.path.exists("data/trajetoria_estimada.csv") else "trajetoria_estimada.csv"
    return pd.read_csv(caminho)

def obter_pose_mais_proxima(ts_lidar, df_trajetoria):
    idx = df_trajetoria['timestamp'].searchsorted(ts_lidar)
    if idx == 0: return df_trajetoria.iloc[0]
    if idx >= len(df_trajetoria): return df_trajetoria.iloc[-1]
    antes = df_trajetoria.iloc[idx - 1]
    depois = df_trajetoria.iloc[idx]
    if abs(ts_lidar - antes['timestamp']) < abs(ts_lidar - depois['timestamp']):
        return antes
    else:
        return depois

def projetar_nuvem(arquivo_lidar, pose):
    print(f"Scan {os.path.basename(arquivo_lidar)} | X: {pose['x']:.6f}, Y: {pose['y']:.6f}")

# Caminho raiz onde os arquivos estão
caminho_base = "log_volta_da_ufes_20230522.txt_lidar"
arquivos_lidar = glob.glob(os.path.join(caminho_base, "**/*.pointcloud"), recursive=True)

df_trajetoria = carregar_trajetoria()

for arquivo in arquivos_lidar[:10]:
    try:
        ts_lidar = float(os.path.basename(arquivo).replace('.pointcloud', ''))
        pose = obter_pose_mais_proxima(ts_lidar, df_trajetoria)
        projetar_nuvem(arquivo, pose)
    except Exception as e:
        print(f"Erro ao processar {arquivo}: {e}")
