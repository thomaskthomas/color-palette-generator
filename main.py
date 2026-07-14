from PIL import Image
import numpy as np
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
rgb_array=np.array(rgb_img)
print(rgb_array)
#flatten
pixels=rgb_array.reshape(-1,3)