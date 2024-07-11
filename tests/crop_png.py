from PIL import Image
png = Image.open("edgesASCII.png")
for i in range(5):
    png.crop((0 + i*8, 0, 8 + i*8, 8)).save(str(i)+".png")