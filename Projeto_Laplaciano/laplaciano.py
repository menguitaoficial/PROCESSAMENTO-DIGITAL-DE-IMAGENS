import cv2
import numpy as np

# Ler imagem
img = cv2.imread("coins.jpg")

# Converter para float
img_float = img.astype(np.float32)

# Aplicar Laplaciano
laplaciano = cv2.Laplacian(img_float, cv2.CV_32F, ksize=3)

# Melhorar nitidez
nitida = img_float - 0.7 * laplaciano

# Limitar valores
nitida = np.clip(nitida, 0, 255).astype(np.uint8)

# Salvar resultado
cv2.imwrite("coins_laplaciano.png", nitida)

# Mostrar imagens
cv2.imshow("Original", img)
cv2.imshow("Nitida", nitida)

cv2.waitKey(0)
cv2.destroyAllWindows()