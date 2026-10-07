import cv2
import numpy as np
import os

print("Diretório atual:", os.getcwd())

# Ler imagem
img = cv2.imread("lente.png")

# Verificar se a imagem foi carregada
if img is None:
    print("Erro: imagem não encontrada!")
    exit()

# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

# Canny
canny = cv2.Canny(gray, 50, 150)

# Laplaciano
laplace = cv2.Laplacian(gray, cv2.CV_64F)
laplace = cv2.convertScaleAbs(laplace)

# Salvar resultados
cv2.imwrite("sobel.png", sobel)
cv2.imwrite("canny.png", canny)
cv2.imwrite("laplace.png", laplace)

# Mostrar imagens
cv2.imshow("Original", img)
cv2.imshow("Sobel", sobel)
cv2.imshow("Canny", canny)
cv2.imshow("Laplace", laplace)

cv2.waitKey(0)
cv2.destroyAllWindows()