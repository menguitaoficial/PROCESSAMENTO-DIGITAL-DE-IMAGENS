import cv2
import numpy as np
import os

print("Diretório atual:", os.getcwd())

# Ler imagem
img = cv2.imread("lagarto.jpg")

# Verificar se carregou corretamente
if img is None:
    print("Erro: imagem não encontrada!")
    exit()

# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Converter para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Canal V (brilho)
v = hsv[:, :, 2]

# =====================
# SOBEL - ESCALA DE CINZA
# =====================
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_gray = cv2.magnitude(sobelx, sobely)
sobel_gray = cv2.convertScaleAbs(sobel_gray)

# =====================
# CANNY - ESCALA DE CINZA
# =====================
canny_gray = cv2.Canny(gray, 50, 150)

# =====================
# LAPLACIANO - ESCALA DE CINZA
# =====================
laplace_gray = cv2.Laplacian(gray, cv2.CV_64F)
laplace_gray = cv2.convertScaleAbs(laplace_gray)

# =====================
# SOBEL - HSV (Canal V)
# =====================
sobelx_v = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)
sobely_v = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)
sobel_hsv = cv2.magnitude(sobelx_v, sobely_v)
sobel_hsv = cv2.convertScaleAbs(sobel_hsv)

# =====================
# CANNY - HSV (Canal V)
# =====================
canny_hsv = cv2.Canny(v, 50, 150)

# =====================
# LAPLACIANO - HSV (Canal V)
# =====================
laplace_hsv = cv2.Laplacian(v, cv2.CV_64F)
laplace_hsv = cv2.convertScaleAbs(laplace_hsv)

# Salvar resultados
cv2.imwrite("sobel_gray.png", sobel_gray)
cv2.imwrite("canny_gray.png", canny_gray)
cv2.imwrite("laplace_gray.png", laplace_gray)

cv2.imwrite("sobel_hsv.png", sobel_hsv)
cv2.imwrite("canny_hsv.png", canny_hsv)
cv2.imwrite("laplace_hsv.png", laplace_hsv)

# Exibir imagens
cv2.imshow("Original", img)

cv2.imshow("Sobel Gray", sobel_gray)
cv2.imshow("Canny Gray", canny_gray)
cv2.imshow("Laplace Gray", laplace_gray)

cv2.imshow("Sobel HSV", sobel_hsv)
cv2.imshow("Canny HSV", canny_hsv)
cv2.imshow("Laplace HSV", laplace_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()