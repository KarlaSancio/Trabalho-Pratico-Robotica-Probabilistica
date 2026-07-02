import pandas as pd
import numpy as np
import os
import glob

def carregar_trajetoria():
    caminho = "data/trajetoria_estimada.csv" if os.path.exists("data/trajetoria_estimada.csv") else "trajetoria_estimada.csv"
    return pd.read_csv(caminho)

def obter_pose(ts_lidar, df_trajetoria):
    idx = df_trajetoria['timestamp'].searchsorted(ts_lidar)
    if idx == 0: return df_trajetoria.iloc[0]
    if idx >= len(df_trajetoria): return df_trajetoria.iloc[-1]
    antes, depois = df_trajetoria.iloc[idx - 1], df_trajetoria.iloc[idx]
    return antes if abs(ts_lidar - antes['timestamp']) < abs(ts_lidar - depois['timestamp']) else depois

def processar_e_salvar(arquivo_lidar, pose):
    # Leitura robusta
    raw_data = np.fromfile(arquivo_lidar, dtype=np.float32)
    
    # Ajuste dinâmico: descarta o que não completar uma linha de 4 elementos
    n_pontos = len(raw_data) // 4
    dados = raw_data[:n_pontos * 4].reshape(-1, 4)
    
    x_local, y_local = dados[:, 0], dados[:, 1]
    
    # Rotação e Translação
    theta = pose['theta']
    x_global = (x_local * np.cos(theta)) - (y_local * np.sin(theta)) + pose['x']
    y_global = (x_local * np.sin(theta)) + (y_local * np.cos(theta)) + pose['y']
    
    os.makedirs("data/processed", exist_ok=True)
    nome_saida = os.path.join("data/processed", os.path.basename(arquivo_lidar).replace('.pointcloud', '.csv'))
    pd.DataFrame({'x': x_global, 'y': y_global}).to_csv(nome_saida, index=False)
    print(f"Processado: {nome_saida} ({n_pontos} pontos)")

caminho_base = "log_volta_da_ufes_20230522.txt_lidar"
arquivos = glob.glob(os.path.join(caminho_base, "**/*.pointcloud"), recursive=True)
df_traj = carregar_trajetoria()

for arquivo in arquivos[:20]: # Aumentamos para 20 para testar a robustez
    try:
        ts = float(os.path.basename(arquivo).replace('.pointcloud', ''))
        processar_e_salvar(arquivo, obter_pose(ts, df_traj))
    except Exception as e:
        print(f"Erro ao processar {arquivo}: {e}")
