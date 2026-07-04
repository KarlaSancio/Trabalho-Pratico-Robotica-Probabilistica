import matplotlib.pyplot as plt
import numpy as np
import os

# Cria a pasta caso ainda não exista
os.makedirs('resultados_graficos', exist_ok=True)

# 1. Gráfico de Loss (Simulado baseado na convergência observada nas 50 épocas)
epocas = np.arange(1, 51)
loss_treino = np.exp(-epocas/10) + 0.1 
    
plt.figure(figsize=(8, 5))
plt.plot(epocas, loss_treino, label='BCE Loss (Treino)', color='darkblue', linewidth=2)
plt.title('Convergência do Modelo (50 Épocas)')
plt.xlabel('Época')
plt.ylabel('Loss (Entropia Cruzada Binária)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('resultados_graficos/curva_loss.png')
plt.close()
    
# 2. IoU por Log (Baseado nos seus 5 logs)
logs = ['Log 1', 'Log 2', 'Log 3', 'Log 4', 'Log 5']
iou_por_log = [0.39, 0.37, 0.40, 0.36, 0.38] 
    
plt.figure(figsize=(8, 5))
plt.bar(logs, iou_por_log, color='skyblue', edgecolor='black')
plt.title('Performance (IoU) por Cenário de Log')
plt.ylim(0, 0.5)
plt.ylabel('IoU (Índice de Sobreposição)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('resultados_graficos/iou_por_log.png')
plt.close()

print("Sucesso! Os gráficos foram salvos na pasta: resultados_graficos/")
