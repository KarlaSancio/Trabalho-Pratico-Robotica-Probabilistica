import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from model import MotionNetSimplificado

def carregar_pontos_binarios(caminho):
    dados = np.fromfile(caminho, dtype=np.float32)
    tamanho = len(dados)
    if tamanho == 0: return np.zeros((0, 2))
    
    if tamanho % 5 == 0 and tamanho % 4 != 0: colunas = 5
    elif tamanho % 4 == 0 and tamanho % 5 != 0: colunas = 4
    elif tamanho % 20 == 0: colunas = 4
    else: return np.zeros((0, 2))
        
    return dados.reshape(-1, colunas)[:, :2] 

def rasterizar(pontos_xy, tamanho_grid=200, limite=20.0):
    grid = np.zeros((tamanho_grid, tamanho_grid), dtype=np.float32)
    if len(pontos_xy) == 0: return grid
        
    x, y = pontos_xy[:, 0], pontos_xy[:, 1]
    mask = np.isfinite(x) & np.isfinite(y) & (x >= -limite) & (x <= limite) & (y >= -limite) & (y <= limite)
    x, y = x[mask], y[mask]
    
    x_idx = ((x + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    y_idx = ((y + limite) / (2 * limite) * (tamanho_grid - 1)).astype(int)
    
    grid[x_idx, y_idx] = 1.0
    return grid

def gerar_visuais():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = MotionNetSimplificado(num_frames_entrada=4).to(device)
    modelo.load_state_dict(torch.load("models/motionnet_ufes.pth", map_location=device, weights_only=True))
    modelo.eval()

    pastas_lidar = sorted(glob.glob("*_lidar"))
    limiar = 0.15

    for pasta_log in pastas_lidar:
        nome_base = pasta_log.replace('_lidar', '')
        arquivos_pcd = sorted(glob.glob(os.path.join(pasta_log, "**", "*.pointcloud"), recursive=True))
        
        # Precisamos de pelo menos 5 frames para formar uma sequência
        if len(arquivos_pcd) < 5:
            continue
            
        print(f"-> Gerando visuais para: {nome_base}...")
        pasta_saida = f"resultados_visuais/{nome_base}"
        os.makedirs(pasta_saida, exist_ok=True)
        
        # 1. Montar a sequência de 5 frames (Formato: 5 frames, 1 canal, 200x200)
        sequencia = np.zeros((5, 1, 200, 200), dtype=np.float32)
        for i in range(5):
            pontos = carregar_pontos_binarios(arquivos_pcd[i])
            sequencia[i, 0] = rasterizar(pontos)
            
        # 2. Salvar visualizacao_sequencia.png
        fig_seq, axes_seq = plt.subplots(1, 5, figsize=(25, 5))
        for i in range(5):
            axes_seq[i].imshow(sequencia[i, 0], cmap='gray')
            axes_seq[i].set_title(f"Frame {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "visualizacao_sequencia.png"))
        plt.close(fig_seq)
        
        # 3. Rodar Inferência e salvar resultado_inferencia.png
        entrada = torch.tensor(sequencia[:4]).unsqueeze(0).to(device) # Pega t=0,1,2,3
        alvo_real = sequencia[4, 0] # Pega t=4 para comparar
        
        with torch.no_grad():
            predicao = modelo(entrada).squeeze().cpu().numpy()
            
        predicao_nitida = (predicao > limiar).astype(float)
        
        fig_inf, axes_inf = plt.subplots(1, 4, figsize=(24, 6))
        axes_inf[0].imshow(sequencia[3, 0], cmap='gray')
        axes_inf[0].set_title("Último Frame (t=3)")
        
        axes_inf[1].imshow(alvo_real, cmap='gray')
        axes_inf[1].set_title("Realidade (t=4)")
        
        axes_inf[2].imshow(predicao, cmap='inferno')
        axes_inf[2].set_title("Incerteza da IA (Probabilidade)")
        
        axes_inf[3].imshow(predicao_nitida, cmap='gray')
        axes_inf[3].set_title(f"Predição Nítida (Limiar > {limiar})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "resultado_inferencia.png"))
        plt.close(fig_inf)

    print("\nTodas as imagens geradas e salvas com sucesso em resultados_visuais/ !")

if __name__ == "__main__":
    gerar_visuais()
