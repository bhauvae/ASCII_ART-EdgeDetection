from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("ModernDOS4378x8.ttf")
print(font.getbbox("@"))
