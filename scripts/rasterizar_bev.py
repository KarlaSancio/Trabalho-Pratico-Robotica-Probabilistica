import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def rasterizar(df, tamanho_grid=200, limite=20.0):
    # Cria uma matriz de zeros
    grid = np.zeros((tamanho_grid, tamanho_grid), dtype=np.float32)
    
    # Normaliza coordenadas para o intervalo [0, tamanho_grid]
    # Limite é a distância máxima (ex: 20m para cada lado)
    x_idx = ((df['x'] + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    y_idx = ((df['y'] + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    
    # Filtra pontos que estão fora do limite
    mask = (x_idx >= 0) & (x_idx < tamanho_grid) & (y_idx >= 0) & (y_idx < tamanho_grid)
    x_idx = x_idx[mask]
    y_idx = y_idx[mask]
    
    # Preenche o grid
    grid[x_idx, y_idx] = 1.0
    return grid

# Processar um exemplo
arquivos = sorted(glob.glob("data/processed/*.csv"))
if arquivos:
    df = pd.read_csv(arquivos[0])
    grid = rasterizar(df)
    
    # Salvar a imagem do grid para conferência
    plt.imsave("data/exemplo_bev.png", grid, cmap='gray')
    print(f"Grid BEV gerado com sucesso: data/exemplo_bev.png")
    print(f"Formato da matriz: {grid.shape}")
