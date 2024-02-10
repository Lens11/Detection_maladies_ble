
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np


def find_dominant_green(image):
    # Convertir l'image en numpy array
    image_np = np.array(image)

    # Extraire les canaux R, G, B
    R, G, B = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]

    # Trouver les pixels principalement verts (simplification)
    green_pixels = (G > R) & (G > B)
    green_indices = np.argwhere(green_pixels)

    # Trouver le centre des pixels verts
    center_of_green = green_indices.mean(axis=0).astype(int)
    return tuple(center_of_green)

def crop_and_save_images(image, output_base_name, center, width=700, height=900):
    for i in range(5):
        left = max(center[1] - width // 2, 0)
        top = max(center[0] - (height * (5 - i)) // 2, 0)
        right = min(left + width, image.width)
        bottom = min(top + height, image.height)

        cropped_image = image.crop((left, top, right, bottom))
        cropped_image.save(f"{output_base_name}_{i}.png")
        

def plot_color_histogram(image):
    # Convertir l'image en RGB si elle est en mode différent
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Obtenir les données des couleurs
    r, g, b = image.split()
    r_data = r.getdata()
    g_data = g.getdata()
    b_data = b.getdata()

    # Afficher l'histogramme pour chaque canal de couleur
    plt.figure(figsize=(10, 4))
    plt.hist(r_data, bins=256, color='red', alpha=0.5, label='Red')
    plt.hist(g_data, bins=256, color='green', alpha=0.5, label='Green')
    plt.hist(b_data, bins=256, color='blue', alpha=0.5, label='Blue')
    plt.legend()
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('Color Histogram')
    plt.show()
    print('ok ?')
    



# Augmenter la limite de taille d'image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Enlève la limite de la taille de l'image


# Utilisez cette partie pour exécuter le script avec votre image
image = Image.open('test.jpg')

#center_of_green = find_dominant_green(image)
r, g, b = image.split()
r_data = r.getdata()
g_data = g.getdata()
b_data = b.getdata()

    # Afficher l'histogramme pour chaque canal de couleur
plt.figure(figsize=(10, 4))
#plt.hist(r_data, bins=256, color='red', alpha=0.5, label='Red')
# plt.hist(g_data, bins=256, color='green', alpha=0.5, label='Green')
# plt.hist(b_data, bins=256, color='blue', alpha=0.5, label='Blue')
#plt.legend()
x = np.linspace(0,10,10)
plt.plot(x,x**2)
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.title('Color Histogram')
plt.show()
print('ok ?')

#plot_color_histogram(image)
#crop_and_save_images(image, "output_image", center_of_green)








