import torch
import numpy as np
from model import MotionNetSimplificado
from dataset import MotionNetDataset

def calcular_metricas():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carrega o modelo com os pesos treinados por 50 épocas
    modelo = MotionNetSimplificado(num_frames_entrada=4).to(device)
    modelo.load_state_dict(torch.load("models/motionnet_ufes.pth", map_location=device, weights_only=True))
    modelo.eval()
    
    dataset = MotionNetDataset()
    
    # O limiar de 15% que definimos visualmente
    limiar = 0.15
    
    tp_total = 0  # Verdadeiros Positivos
    fp_total = 0  # Falsos Positivos
    fn_total = 0  # Falsos Negativos
    
    print(f"Calculando métricas em todo o conjunto de dados usando Limiar de {limiar}...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            amostra = dataset[i]
            entrada = amostra[:4, :, :, :].unsqueeze(0).to(device)
            alvo_real = amostra[4, 0, :, :].numpy()
            
            # Pega a predição da rede
            predicao = modelo(entrada).squeeze().cpu().numpy()
            
            # Aplica o filtro binário (transforma probabilidade em 0 ou 1)
            predicao_binaria = (predicao > limiar).astype(float)
            
            # Achata as matrizes para comparar pixel por pixel
            pred_flat = predicao_binaria.flatten()
            alvo_flat = alvo_real.flatten()
            
            # Cálculos de interseção
            tp_total += np.sum((pred_flat == 1) & (alvo_flat == 1))
            fp_total += np.sum((pred_flat == 1) & (alvo_flat == 0))
            fn_total += np.sum((pred_flat == 0) & (alvo_flat == 1))
            
    # Matemática final
    precisao = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0
    
    print("\n" + "="*45)
    print("      MÉTRICAS QUANTITATIVAS DA IA")
    print("="*45)
    print(f" Precisão (Precision) : {precisao:.4f}  ({precisao*100:.2f}%)")
    print(f" Revocação (Recall)   : {recall:.4f}  ({recall*100:.2f}%)")
    print(f" IoU                  : {iou:.4f}  ({iou*100:.2f}%)")
    print("="*45)
    print("\n-> Estes são os valores exatos para a tabela do seu relatório!")

if __name__ == "__main__":
    calcular_metricas()
