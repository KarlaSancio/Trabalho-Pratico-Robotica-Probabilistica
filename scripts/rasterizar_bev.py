import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def carregar_pontos_binarios(caminho):
    """Lê o arquivo .pointcloud original do CARMEN-LCAD."""
    dados = np.fromfile(caminho, dtype=np.float32)
    tamanho = len(dados)
    
    if tamanho == 0:
        return np.zeros((0, 2))
        
    # Detecção dinâmica: o CARMEN varia entre 4 ou 5 atributos ao longo dos meses
    if tamanho % 5 == 0 and tamanho % 4 != 0:
        colunas = 5
    elif tamanho % 4 == 0 and tamanho % 5 != 0:
        colunas = 4
    elif tamanho % 20 == 0: 
        # Empate matemático, assumimos o padrão mais antigo de 4 colunas
        colunas = 4
    else:
        # Dados do frame corrompidos, devolve vazio para não travar o lote
        return np.zeros((0, 2))
        
    pontos = dados.reshape(-1, colunas)
    return pontos[:, :2] 

def rasterizar(pontos_xy, tamanho_grid=200, limite=20.0):
    """Converte os pontos contínuos para a grade discreta do BEV."""
    grid = np.zeros((tamanho_grid, tamanho_grid), dtype=np.float32)
    
    if len(pontos_xy) == 0:
        return grid
        
    x = pontos_xy[:, 0]
    y = pontos_xy[:, 1]
    
    # Filtro de Robustez
    mask_validos = np.isfinite(x) & np.isfinite(y) & (x >= -limite) & (x <= limite) & (y >= -limite) & (y <= limite)
    x = x[mask_validos]
    y = y[mask_validos]
    
    x_idx = ((x + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    y_idx = ((y + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    
    grid[x_idx, y_idx] = 1.0
    return grid

# Encontra todas as pastas geradas a partir dos logs (terminadas em _lidar)
pastas_lidar = sorted(glob.glob("*_lidar"))

if not pastas_lidar:
    print("Aviso: Nenhuma pasta '_lidar' encontrada na raiz.")
else:
    print(f"Encontrados {len(pastas_lidar)} logs extraídos para processar.")
    
    for pasta_log in pastas_lidar:
        nome_base = pasta_log.replace('_lidar', '')
        arquivos_pcd = sorted(glob.glob(os.path.join(pasta_log, "**", "*.pointcloud"), recursive=True))
        
        if not arquivos_pcd:
            continue
            
        print(f"-> Processando: {nome_base} ({len(arquivos_pcd)} frames encontrados)...")
        
        pasta_saida = f"resultados_visuais/{nome_base}"
        os.makedirs(pasta_saida, exist_ok=True)
        
        primeiro_frame = arquivos_pcd[0]
        pontos_xy = carregar_pontos_binarios(primeiro_frame)
        grid = rasterizar(pontos_xy)
        
        caminho_imagem = os.path.join(pasta_saida, "mapa_bev.png")
        plt.imsave(caminho_imagem, grid, cmap='gray')
        
        caminho_npy = os.path.join(pasta_saida, "mapa_bev.npy")
        np.save(caminho_npy, grid)
        
    print("\nProcessamento em lote concluído com sucesso!")
