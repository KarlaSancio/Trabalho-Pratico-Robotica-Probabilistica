# Integração da MotionNet ao CARMEN-LCAD para Percepção e Predição de Movimento Baseado em LiDAR

**Dupla:** Karla Sancio e Matheus Lopes

## Dependências e Preparação do Ambiente

Recomenda-se a utilização de um ambiente virtual (ex: `venv` ou `conda`) para isolar as dependências do projeto. 

As bibliotecas necessárias para a execução dos scripts são:
* `torch` (PyTorch)
* `numpy`
* `matplotlib`

Para instalar as dependências via `pip`:
```bash
pip install torch numpy matplotlib
```

## Organização dos Dados (Logs da IARA)
Para que os scripts funcionem corretamente, os logs brutos da IARA devem ser extraídos na raiz do projeto. O pipeline foi projetado para detectar automaticamente as pastas de saída do LiDAR.

Sua estrutura de diretórios deve ficar assim:
```text
/seu_repositorio
│
├── log_volta_da_ufes_20230522_lidar/      # Pasta contendo os arquivos .pointcloud
├── log_volta_da_ufes_20230617-1_lidar/    # Pasta contendo os arquivos .pointcloud
│
├── scripts/
│   ├── model.py
│   ├── dataset.py
│   ├── rasterizar_bev.py
│   ├── treinar.py
│   ├── calcular_metricas.py
│   └── gerar_visuais_finais.py
│
└── README.md
```
## Execução do Pipeline
Siga a ordem abaixo para reproduzir os resultados e as métricas do projeto:

### 1. Pré-processamento e Rasterização
Este script vasculha as pastas `_lidar`, lida automaticamente com variações de calibração do sensor (4 ou 5 atributos por feixe), filtra anomalias e converte as nuvens de pontos contínuas em grades de ocupação BEV (200x200).

```bash
python scripts/rasterizar_bev.py
```

### 2. Treinamento do Modelo
Treina a rede neural (Encoder-Decoder com Resize Convolutions para evitar artefatos quadriculados) por 50 épocas utilizando os tensores históricos para prever o frame futuro. Os pesos serão salvos na pasta `models/`.
```bash
python scripts/treinar.py
```

### 3. Avaliação Quantitativa (Métricas)
Calcula e exibe no terminal as métricas de performance do modelo treinado utilizando um limiar de probabilidade de 15% ($\tau = 0.15$): Precisão, Revocação (Recall) e IoU.
```bash
python scripts/calcular_metricas.py
```
### 4. Geração de Resultados Visuais
Extrai sequências de 5 frames de cada log processado, executa a inferência e salva os mapas probabilísticos (Incerteza da IA) e a predição espacial nítida. Todas as imagens são salvas automaticamente na pasta `resultados_visuais/`.
```bash
python scripts/gerar_visuais_finais.py
```

## Detalhes da Arquitetura
A rede foi modificada em relação ao conceito tradicional de convoluções transpostas. Para contornar o problema de *Checkerboard Artifact*s no Decoder durante o redimensionamento das matrizes esparsas do LiDAR, implementou-se blocos de Resize Convolution (interpolação espacial seguida de convolução de passo unitário), garantindo a convergência espacial contínua e estimativas de incerteza confiáveis nas bordas dos obstáculos.

## Documentação Técnica
O relatório detalhado sobre o desenvolvimento, integração e análise probabilística do pipeline de percepção preditiva está disponível na raiz do projeto:

*   **Relatório Final:** [Relatorio_Integracao_Motion_Net.pdf](./Relatorio_Integracao_Motion_Net.pdf)