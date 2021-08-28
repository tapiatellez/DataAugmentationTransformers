import matplotlib.pyplot as plt

#   Spam data
percent = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
class_plain_spam = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.85, 0.5, 0.5, 0.85, 0.85]
class_augmented_spam = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.98, 0.72, 0.93, 0.86, 0.89]
#   IMDB data
class_plain_IMDB = [0, 0.52, 0.52, 0.52, 0.52, 0.52, 0.61, 0.60, 0.66, 0.70, 0.50]
class_augmented_IMDB = [0, 0.54, 0.53, 0.51, 0.56, 0.60, 0.67, 0.64, 0.67, 0.70, 0.66]
#   Sentence Type Data
class_plain_ST = [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.46, 0.40, 0.40, 0.40, 0.40 ]
class_augmented_ST = [0.0, 0.33, 0.33, 0.33, 0.47, 0.44, 0.34, 0.48, 0.51, 0.46, 0.45]
#   SST2
class_plain_SST2 = [0.0, 0.44, 0.60, 0.47, 0.57, 0.61, 0.62, 0.59, 0.62, 0.66, 0.68]
class_augmented_SST2 = [0.0, 0.53, 0.56, 0.55, 0.49, 0.53, 0.67, 0.54, 0.57, 0.64, 0.71]
#   Create figure
fig, ((g1, g2), (g3, g4)) = plt.subplots(2,2)
#   Plot with differently-colored markers
#   Plot for Spam
g1.plot(percent, class_plain_spam, 'o-', label = 'Plain')
g1.plot(percent, class_augmented_spam, '.-', label = 'Augmented')
g1.legend(loc = 'lower right')
g1.set_xlabel('Percent of Dataset (%)')
g1.set_ylabel('Accuracy')
g1.set_title('Spam (N=500)')
#   Plot for IMDB
g2.plot(percent, class_plain_IMDB, 'o-', label = 'Plain')
g2.plot(percent, class_augmented_IMDB, '.-', label = 'Augmented')
g2.legend(loc = 'lower right')
g2.set_xlabel('Percent of Dataset (%)')
g2.set_ylabel('Accuracy')
g2.set_title('IMDB (N=500)')
#   Plot for SST2
g3.plot(percent, class_plain_SST2, 'o-', label = 'Plain')
g3.plot(percent, class_augmented_SST2, '.-', label = 'Augmented')
g3.legend(loc = 'lower right')
g3.set_xlabel('Percent of Dataset (%)')
g3.set_ylabel('Accuracy')
g3.set_title('SST2 (N=500)')
#   Plot for ST
g4.plot(percent, class_plain_ST, 'o-', label = 'Plain')
g4.plot(percent, class_augmented_ST, '.-', label = 'Augmented')
g4.legend(loc = 'lower right')
g4.set_xlabel('Percent of Dataset (%)')
g4.set_ylabel('Accuracy')
g4.set_title('ST (N=500)')
#   Set figure
fig.tight_layout()
plt.show()
