import pandas as pd
import matplotlib.pyplot as plt
import sys

def plotar_trajetoria(arquivo_csv):
    try:
        # Carrega os dados
        df = pd.read_csv(arquivo_csv)
        
        # Configura o gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(df['x'], df['y'], label='Trajetória IARA', color='blue', linewidth=1.5)
        
        # Marca o início e o fim
        plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', label='Início', zorder=5)
        plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', label='Fim', zorder=5)
        
        # Estilização
        plt.xlabel('X (metros)')
        plt.ylabel('Y (metros)')
        plt.title('Trajetória Estimada do Veículo IARA')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        
        # Salva o gráfico
        output_file = 'data/trajetoria_plot.png'
        plt.savefig(output_file, dpi=300)
        print(f"Gráfico salvo com sucesso em: {output_file}")
        plt.show()
        
    except Exception as e:
        print(f"Erro ao plotar: {e}")

if __name__ == "__main__":
    arquivo = "data/trajetoria_estimada.csv"
    plotar_trajetoria(arquivo)
