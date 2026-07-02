import pandas as pd
import numpy as np
import os
import glob

def rasterizar(df, tamanho_grid=200, limite=20.0):
    grid = np.zeros((tamanho_grid, tamanho_grid), dtype=np.float32)
    x_idx = ((df['x'] + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    y_idx = ((df['y'] + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    mask = (x_idx >= 0) & (x_idx < tamanho_grid) & (y_idx >= 0) & (y_idx < tamanho_grid)
    grid[x_idx[mask], y_idx[mask]] = 1.0
    return grid

# Configurações
arquivos = sorted(glob.glob("data/processed/*.csv"))
tamanho_sequencia = 5
os.makedirs("data/sequences", exist_ok=True)

# Gera sequências deslizantes
for i in range(len(arquivos) - tamanho_sequencia):
    sequencia = []
    for j in range(tamanho_sequencia):
        df = pd.read_csv(arquivos[i + j])
        sequencia.append(rasterizar(df))
    
    # Salva como um tensor (Time, H, W)
    np.save(f"data/sequences/seq_{i:04d}.npy", np.array(sequencia))
    if i % 10 == 0:
        print(f"Sequência {i} salva em data/sequences/seq_{i:04d}.npy")

print("Dataset de sequências pronto!")
