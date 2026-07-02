import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def plotar_acumulado():
    arquivos = sorted(glob.glob("data/processed/*.csv"))[:100]
    plt.figure(figsize=(8, 8))
    
    for arquivo in arquivos:
        df = pd.read_csv(arquivo)
        # Filtra apenas dados válidos (remove NaN ou Infinitos)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Plota os pontos
        plt.scatter(df['x'], df['y'], s=0.1, alpha=0.1, c='blue')
    
    # Define limites manuais para evitar escala astronômica (ex: +/- 20 metros)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    
    plt.title("Visualização da Trajetória Acumulada (Filtro Aplicado)")
    plt.grid(True)
    plt.savefig("trajetoria_global_limpa.png")
    print("Gráfico gerado: trajetoria_global_limpa.png")

plotar_acumulado()
