import numpy as np
import csv

def integrar_trajetoria(arquivo_log, arquivo_saida):
    trajetoria = []
    x, y, theta = 0.0, 0.0, 0.0
    last_ts = None
    
    with open(arquivo_log, 'r') as f:
        for linha in f:
            if "ROBOTVELOCITY_ACK" in linha:
                partes = linha.split()
                # Procurando o timestamp que geralmente está antes do nome do processo
                # Exemplo: ... 1684869977.656972 ford_escape_hybrid@relampago 0.005571
                try:
                    # O timestamp está na penúltima posição antes do nome do processo
                    # Vamos pegar o valor que for float antes do nome do processo
                    ts = float(partes[-3])
                    tv = float(partes[1])
                    rv = float(partes[2])
                    
                    if last_ts is not None:
                        dt = ts - last_ts
                        theta += rv * dt
                        x += tv * np.cos(theta) * dt
                        y += tv * np.sin(theta) * dt
                    
                    trajetoria.append([ts, x, y, theta])
                    last_ts = ts
                except (ValueError, IndexError):
                    continue
    
    with open(arquivo_saida, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'theta'])
        writer.writerows(trajetoria)
    
    return trajetoria

log_input = "log_volta_da_ufes_20230522.txt" #[cite: 1]
csv_output = "data/trajetoria_estimada.csv"
trajetoria = integrar_trajetoria(log_input, csv_output)
print(f"Sucesso! Trajetória salva em {csv_output} com {len(trajetoria)} registros.")
