import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem = np.zeros((300, 500), dtype=np.uint8)

cv2.rectangle(imagem, (0, 0), (500, 300), 50, -1)
cv2.rectangle(imagem, (150, 100), (350, 200), 180, -1)

ruido = np.zeros(imagem.shape, np.int16)
cv2.randn(ruido, 0, 20)
imagem = cv2.add(imagem, ruido, dtype=cv2.CV_8UC1)

hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])
hist_suavizado = cv2.GaussianBlur(hist.astype(np.float32), (5, 5), 0)
pico1_idx = np.argmax(hist_suavizado)

hist_temp = np.copy(hist_suavizado)

zero_out_range = 40 

start_zero = max(0, pico1_idx - zero_out_range)
end_zero = min(len(hist_temp), pico1_idx + zero_out_range)

hist_temp[start_zero:end_zero] = 0

pico2_idx = np.argmax(hist_temp)

if pico1_idx > pico2_idx:
    pico1_idx, pico2_idx = pico2_idx, pico1_idx

min_peak_separation = 10

if abs(pico1_idx - pico2_idx) < min_peak_separation:
    print(f"Aviso: Os picos estão muito próximos ({pico1_idx} e {pico2_idx}).")
    print("A distribuição bimodal pode não ser clara ou o limiar do vale pode não ser ideal.")
    limiar_vale = (pico1_idx + pico2_idx) // 2
else:
    vale_idx_relative = np.argmin(hist_suavizado[pico1_idx:pico2_idx])
    limiar_vale = pico1_idx + vale_idx_relative

print(f"Pico 1 encontrado em: {pico1_idx}")
print(f"Pico 2 encontrado em: {pico2_idx}")
print(f"✨ Limiar (Vale) encontrado em: {limiar_vale}")

_, imagem_segmentada = cv2.threshold(imagem, limiar_vale, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(hist, color='blue')
plt.title('Histograma e Limiar do Vale')
plt.xlabel('Nível de Cinza')
plt.ylabel('Frequência')
plt.axvline(x=limiar_vale, color='r', linestyle='--', label=f'Limiar = {limiar_vale}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 3, 3)
plt.imshow(imagem_segmentada, cmap='gray')
plt.title('Imagem Segmentada')
plt.axis('off')

plt.tight_layout()
plt.show()