import pandas as pd
import numpy as np

# Carrega a primeira linha da trajetória
df = pd.read_csv("data/trajetoria_estimada.csv")
pose = df.iloc[1] # Pega a segunda linha (onde o robô começa a andar)

print(f"Pose usada: X={pose['x']}, Y={pose['y']}, Theta={pose['theta']}")

# Exemplo de um ponto local (lidar)
x_local, y_local = 1.0, 1.0 

# Aplica a transformação
x_global = (x_local * np.cos(pose['theta'])) - (y_local * np.sin(pose['theta'])) + pose['x']
y_global = (x_local * np.sin(pose['theta'])) + (y_local * np.cos(pose['theta'])) + pose['y']

print(f"Resultado transformado: X_global={x_global}, Y_global={y_global}")
