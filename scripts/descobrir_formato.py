import numpy as np

arquivo = "log_volta_da_ufes_20230522.txt_lidar/1684870000/1684870400/1684870436.981496.pointcloud"
dados = np.fromfile(arquivo, dtype=np.float32)
tamanho = len(dados)

print(f"Total de números lidos: {tamanho}")

for colunas in range(3, 7):
    if tamanho % colunas == 0:
        print(f"Possível estrutura encontrada: {colunas} colunas por ponto (serão {tamanho // colunas} pontos).")
