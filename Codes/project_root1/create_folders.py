import os

"""
Script para criar a estrutura de pastas do projeto com 4 classes:
Atelectasis, Cardiomegaly, Effusion e Pneumonia.

Estrutura resultante:

project_root/
├── data/
│   ├── train/             # Dados de treino (originais + sintéticos)
│   │   ├── Atelectasis/
│   │   ├── Cardiomegaly/
│   │   ├── Effusion/
│   │   └── Pneumonia/
│   ├── val/               # Validação (somente originais)
│   │   ├── Atelectasis/
│   │   ├── Cardiomegaly/
│   │   ├── Effusion/
│   │   └── Pneumonia/
│   └── test/              # Teste final (somente originais, hold-out)
│       ├── Atelectasis/
│       ├── Cardiomegaly/
│       ├── Effusion/
│       └── Pneumonia/
│
├── generated/             # Imagens sintéticas (PGGANs)
│   └── train/             # Apenas treino (sem val/test)
│       ├── Atelectasis/
│       ├── Cardiomegaly/
│       ├── Effusion/
│       └── Pneumonia/
│
├── Trained_Models/        # Modelos, métricas, CSVs e plots
│   └── logs/
│
└── notebooks/             # Notebooks principais (.ipynb)
"""

# Definição das classes
classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia']

# Criação das pastas principais
os.makedirs('data', exist_ok=True)
os.makedirs('generated/train', exist_ok=True)
os.makedirs('Trained_Models/logs', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# Subpastas de treino, validação e teste
for subset in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join('data', subset, cls), exist_ok=True)

# Subpastas de imagens geradas (só para treino)
for cls in classes:
    os.makedirs(os.path.join('generated/train', cls), exist_ok=True)

print("✅ Estrutura de pastas criada com sucesso!\n")
print("Organize as imagens da seguinte forma:\n")
print("  - data/train/<classe>/ → 70% originais + (opcionalmente) sintéticas mixadas")
print("  - data/val/<classe>/   → 15% originais (validação)")
print("  - data/test/<classe>/  → 15% originais (teste hold-out)")
print("  - generated/train/<classe>/ → sintéticas (para mistura no treino)")
print("\nAgora mova ou copie as imagens para as respectivas subpastas.")
