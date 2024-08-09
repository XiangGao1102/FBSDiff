from PIL import Image

Image.open('../livingroom.png').resize((512, 512)).save('../livingroom.png')