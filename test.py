import os
from PIL import Image
import matplotlib.pyplot as plt

path = 'datasets/CFG/seen_font500_800'
styles = os.listdir(path)
for style in styles:
    style_path = os.path.join(path,style,'哧.png')
    print(style)
    image = Image.open(style_path)
    plt.imshow(image)
    option = input('n to reject')


