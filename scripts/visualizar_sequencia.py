import numpy as np
import matplotlib.pyplot as plt
import glob

# Pega a primeira sequência gerada
seq_files = sorted(glob.glob("data/sequences/*.npy"))
if not seq_files:
    print("Nenhum arquivo .npy encontrado em data/sequences/")
else:
    seq = np.load(seq_files[0])
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(seq[i], cmap='gray')
        axes[i].set_title(f"Frame {i}")
    
    plt.tight_layout()
    plt.savefig("visualizacao_sequencia.png")
    print("Sequência visualizada em: visualizacao_sequencia.png")
