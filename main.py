from PIL import Image

#img = Image.open(r"COLOR_GEN\image.jpg")
img = Image.open(r"image.jpg")
rgb_img = img.convert('RGB')
rgb_img.show()

#downsize pixels
width,height= rgb_img.size
print(width*height)

rgb_img.thumbnail((244,244), Image.Resampling.LANCZOS)
rgb_img.show()

#convert to array