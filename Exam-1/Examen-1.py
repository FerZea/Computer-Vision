import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

# =============================
# 1. Cargar imagen
# =============================
img = cv2.imread("/home/ferzea/Documents/Vision-Computacional/Exam-1/images/camera_man_ruido.png")

if img is None:
    raise IOError("No se pudo cargar la imagen")

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =============================
# 2. Aplicación de filtros
# (mínimo 3, máximo 6)
# =============================

# Filtro 1 (ejemplo)
filtro1 = cv2.GaussianBlur(gray, (3, 3), 0)


# Filtro 2 (ejemplo)
filtro2 = cv2.medianBlur(filtro1, 3)


# Filtro 3 (ejemplo)

kernel = np.ones((3, 3), np.uint8)
filtro3 = cv2.morphologyEx(filtro2, cv2.MORPH_OPEN, kernel)

kernel_sharp = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
filtro4 = cv2.filter2D(filtro3, -1, kernel_sharp)

# Guarda resultado intermedio
cv2.imwrite("images/paso1_filtros.jpg", filtro4)

# =============================arpen (opcional)
# 3. Recorte de la persona
# =============================

# --- Aquí puedes:
# - usar umbralización
# - detección de contornos
# - bounding box
# (solo se deja la estructura)

_, thresh = cv2.threshold(filtro3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contornos, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Seleccionar el contorno más grande
contorno_principal = max(contornos, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(contorno_principal)
recorte = img[y:y+h, x:x+w]

cv2.imwrite("images/paso2_recorte.jpg", recorte)

# =============================
# 4. Rotación de la imagen
# =============================

# Ángulo de rotación (AJUSTAR)
angulo = 10  # <-- aquí debes calcularlo o ajustarlo

(h, w) = recorte.shape[:2]
centro = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
rotada = cv2.warpAffine(recorte, M, (w, h))

cv2.imwrite("images/paso3_rotacion.jpg", rotada)

# =============================
# 5. Mostrar resultados
# =============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Filtros")
plt.imshow(filtro3, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Recorte")
plt.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Rotación")
plt.imshow(cv2.cvtColor(rotada, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.savefig(os.path.join(IMG_DIR, "resultados.png"),
            dpi=300, bbox_inches="tight")
plt.close()