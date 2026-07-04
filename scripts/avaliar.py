import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MotionNetSimplificado
from dataset import MotionNetDataset

def avaliar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo = MotionNetSimplificado(num_frames_entrada=4).to(device)
    modelo.load_state_dict(torch.load("models/motionnet_ufes.pth", map_location=device, weights_only=True))
    modelo.eval()
    
    dataset = MotionNetDataset()
    amostra = dataset[10 if len(dataset) > 10 else 0]
    
    entrada = amostra[:4, :, :, :].unsqueeze(0).to(device) 
    alvo_real = amostra[4, 0, :, :].numpy() 
    
    with torch.no_grad():
        predicao = modelo(entrada)
        
    predicao_np = predicao.squeeze().cpu().numpy()
    
    # --- LIMIAR FIXO ---
    # Revela tudo que tiver mais de 15% de probabilidade de ser um obstáculo
    limiar = 0.15  
    predicao_nitida = (predicao_np > limiar).astype(float)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    axes[0].imshow(amostra[3, 0, :, :].numpy(), cmap='gray')
    axes[0].set_title("Último Frame (t=3)")
    
    axes[1].imshow(alvo_real, cmap='gray')
    axes[1].set_title("Realidade (t=4)")
    
    axes[2].imshow(predicao_np, cmap='inferno')
    axes[2].set_title("Incerteza da IA (Probabilidade)")
    
    axes[3].imshow(predicao_nitida, cmap='gray')
    axes[3].set_title(f"Predição Nítida (Limiar > {limiar})")
    
    plt.tight_layout()
    plt.savefig("resultado_inferencia.png")
    print("Avaliação concluída! Imagem salva como: resultado_inferencia.png")

if __name__ == "__main__":
    avaliar_modelo()
