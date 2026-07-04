import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MotionNetDataset
from model import MotionNetSimplificado
import os

def treinar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50  # <-- Aumentamos para 50 épocas
    batch_size = 8
    lr = 0.001
    
    print(f"Treinando no dispositivo: {device}")
    
    dataset = MotionNetDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    modelo = MotionNetSimplificado(num_frames_entrada=4).to(device)
    otimizador = optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.BCELoss() 
    
    modelo.train()
    for epoch in range(epochs):
        loss_epoch = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            entrada = batch[:, :4, :, :, :]
            alvo = batch[:, 4, :, :, :] 
            
            otimizador.zero_grad()
            predicao = modelo(entrada)
            loss = criterio(predicao, alvo)
            loss.backward()
            otimizador.step()
            loss_epoch += loss.item()
            
        # Imprime a cada 5 épocas para não poluir o terminal
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época [{epoch+1}/{epochs}] - Perda (Loss): {loss_epoch / len(dataloader):.4f}")
    
    os.makedirs("models", exist_ok=True)
    torch.save(modelo.state_dict(), "models/motionnet_ufes.pth")
    print("Treino concluído! Pesos guardados.")

if __name__ == "__main__":
    treinar_modelo()
