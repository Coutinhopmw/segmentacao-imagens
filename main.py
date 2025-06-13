import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Criar uma imagem de exemplo com histograma bimodal ---
# Imagem preta de 300x500 pixels
imagem = np.zeros((300, 500), dtype=np.uint8)

# Adicionar um fundo mais escuro (valores em torno de 50)
# e um objeto mais claro no centro (valores em torno de 180)
cv2.rectangle(imagem, (0, 0), (500, 300), 50, -1)
cv2.rectangle(imagem, (150, 100), (350, 200), 180, -1)

# Adicionar um pouco de ruído gaussiano para tornar o histograma mais realista
ruido = np.zeros(imagem.shape, np.int16)
cv2.randn(ruido, 0, 20) # Ruído com média 0 e desvio padrão 20
imagem = cv2.add(imagem, ruido, dtype=cv2.CV_8UC1)


# --- 2. Calcular o histograma ---
# Parâmetros: imagem, canais, máscara, tamanho, range
hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])


# --- 3. Encontrar os dois picos principais no histograma ---
# Suaviza o histograma para facilitar a detecção de picos
hist_suavizado = cv2.GaussianBlur(hist, (5, 5), 0)

# Encontra o primeiro pico (o mais alto)
pico1_idx = np.argmax(hist_suavizado)

# Zera a região ao redor do primeiro pico para encontrar o segundo
hist_temp = np.copy(hist_suavizado)
cv2.rectangle(hist_temp, (pico1_idx - 30, 0), (pico1_idx + 30, 0), 0, -1)
pico2_idx = np.argmax(hist_temp)

# Garante que pico1 < pico2 para a busca do vale
if pico1_idx > pico2_idx:
    pico1_idx, pico2_idx = pico2_idx, pico1_idx


# --- 4. Encontrar o vale (ponto mínimo) entre os picos ---
# O vale é o ponto de menor frequência no histograma entre os dois picos
vale_idx = pico1_idx + np.argmin(hist_suavizado[pico1_idx:pico2_idx])
limiar_vale = vale_idx

print(f"Pico 1 encontrado em: {pico1_idx}")
print(f"Pico 2 encontrado em: {pico2_idx}")
print(f"✨ Limiar (Vale) encontrado em: {limiar_vale}")


# --- 5. Aplicar o limiar para segmentar a imagem ---
_, imagem_segmentada = cv2.threshold(imagem, limiar_vale, 255, cv2.THRESH_BINARY)


# --- 6. Exibir os resultados ---
plt.figure(figsize=(18, 5))

# Imagem Original
plt.subplot(1, 3, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Histograma e Limiar
plt.subplot(1, 3, 2)
plt.plot(hist, color='blue')
plt.title('Histograma e Limiar do Vale')
plt.xlabel('Nível de Cinza')
plt.ylabel('Frequência')
plt.axvline(x=limiar_vale, color='r', linestyle='--', label=f'Limiar = {limiar_vale}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Imagem Segmentada
plt.subplot(1, 3, 3)
plt.imshow(imagem_segmentada, cmap='gray')
plt.title('Imagem Segmentada')
plt.axis('off')

plt.tight_layout()
plt.show()