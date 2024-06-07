import cv2
import numpy as np

# Création d'une image vide (blanche)
width, height = 400, 400
image = np.zeros((height, width), dtype=np.uint8)

# Définir les paramètres des ellipses
# Ellipse 1 : au centre de l'image
center1 = (200, 200)
axes1 = (100, 50)
angle1 = 0
startAngle1 = 0
endAngle1 = 360
color1 = 255
thickness1 = 2

# Ellipse 2 : en haut à gauche
center2 = (100, 100)
axes2 = (60, 30)
angle2 = 45
startAngle2 = 0
endAngle2 = 360
color2 = 255
thickness2 = 2

# Ellipse 3 : débordant du bord droit de l'image
center3 = (350, 200)
axes3 = (100, 50)
angle3 = 30
startAngle3 = 0
endAngle3 = 360
color3 = 255
thickness3 = 2

# Dessiner les ellipses sur l'image
cv2.ellipse(image, center1, axes1, angle1, startAngle1, endAngle1, color1, thickness1)
cv2.ellipse(image, center2, axes2, angle2, startAngle2, endAngle2, color2, thickness2)
cv2.ellipse(image, center3, axes3, angle3, startAngle3, endAngle3, color3, thickness3)

# Appliquer un flou gaussien pour réduire le bruit
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Utiliser la méthode de détection de contours
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Copier l'image pour dessiner les ellipses détectées
output = image.copy()

# Détection et dessin des ellipses
print(f"{len(contours) = }"
for cont in contours:
    if len(cont) >= 5:  # La fonction fitEllipse requiert au moins 5 points
        ellipse = cv2.fitEllipse(cont)
        cv2.ellipse(output, ellipse, 127, 2)  # Dessiner les ellipses détectées en jaune

# Afficher l'image originale et l'image avec les ellipses détectées
cv2.imshow('ellipses', image)
cv2.imshow('detected ellipses', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sauvegarder l'image avec les ellipses détectées
cv2.imwrite('ellipses_detected.png', output)
