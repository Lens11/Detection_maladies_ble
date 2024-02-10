# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# def load_and_subsample_image(image_path, step=10):
#     # Charger l'image avec OpenCV
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB

#     # Sous-échantillonnage
#     subsampled_img = img[::step, ::step]
#     return subsampled_img

# def plot_2d_color_intensity(image_data):
#     # Calculer la moyenne des intensités pour chaque canal de couleur le long de la hauteur
#     mean_intensities = np.mean(image_data, axis=0)

#     # Tracer le graphique
#     plt.figure(figsize=(10, 4))
#     for i, color in enumerate(['Red', 'Green', 'Blue']):
#         plt.plot(mean_intensities[:, i], label=f'{color} Intensity', color=color.lower())
    
#     plt.xlabel('Width (pixels)')
#     plt.ylabel('Average Color Intensity')
#     plt.title('Average Color Intensity across Image Width')
#     plt.legend()
#     plt.show()

# # Chemin vers votre image
# image_path = 'part_0.png'
# subsampled_image_data = load_and_subsample_image(image_path)

# plot_2d_color_intensity(subsampled_image_data)
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image_and_horizontal_lines(image_path, lines):
    # Charger l'image avec OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dessiner les lignes horizontales sur l'image
    for line in lines:
        cv2.line(img, (0, line), (img.shape[1], line), (255, 0, 0), 20)

    # Afficher l'image avec les lignes
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title('Original Image with Horizontal Lines')
    plt.show()

    # Tracer l'intensité des couleurs pour chaque ligne horizontale
    fig, axes = plt.subplots(len(lines), 1, figsize=(12, 6), sharex=True)

    for index, line in enumerate(lines):
        if line < img.shape[0]:  # Vérifier si la ligne est dans les limites de l'image
            color_data = img[line, :, :]
            #print(f'Color data for line {line}: {color_data[1100:1200]}')
            #axes[index].plot(color_data[:, 0], color='red', label='Red Intensity')
            axes[index].plot(color_data[:, 1], color='green', label='Green Intensity')
            #axes[index].plot(color_data[:, 2], color='blue', label='Blue Intensity')
            axes[index].set_ylabel(f'Line {line}')
            axes[index].legend()

    plt.xlabel('Width (pixels)')
    plt.suptitle('Color Intensity across Horizontal Lines')
    plt.show()

# Chemin vers votre image et lignes horizontales à tracer
image_path = 'part_0.png'
horizontal_lines = [6000, 9000, 12000]

plot_image_and_horizontal_lines(image_path, horizontal_lines)
