import torch
import torch.nn as nn

class MotionNetSimplificado(nn.Module):
    def __init__(self, num_frames_entrada=5):
        super(MotionNetSimplificado, self).__init__()
        
        # Encoder: Mantemos igual (extrai as características espaciais)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_frames_entrada, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder: Substituímos ConvTranspose2d por Upsample + Conv2d (Resize Convolution)
        self.decoder = nn.Sequential(
            # Amplia de 25x25 para 50x50 e suaviza
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Amplia de 50x50 para 100x100 e suaviza
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Amplia de 100x100 para 200x200 (Frame final)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = x.squeeze(2) 
        features = self.encoder(x)
        predicao = self.decoder(features)
        return predicao

if __name__ == "__main__":
    modelo = MotionNetSimplificado()
    tensor_teste = torch.randn(2, 5, 1, 200, 200)
    saida = modelo(tensor_teste)
    print(f"Formato da saída do modelo atualizado (Resize Conv): {saida.shape}")
