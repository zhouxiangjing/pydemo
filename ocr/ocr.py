
# conda install -c simonflueckiger tesserocr

from PIL import Image
# from pytesseract import image_to_string
import tesserocr

print(tesserocr.get_languages())

#新建Image对象
image = Image.open("C:/Users/86529/Desktop/123.jpg")
# 转为灰度图片
image_grey = image.convert('L')
# image_grey.show()
# 二值化
table = []
for i in range(256):
    if i < 140:
        table.append(0)
    else:
        table.append(1)
image_bi = image_grey.point(table, '1')
# 识别验证码
verify_code = tesserocr.image_to_text(image_bi)
#
# #调用tesserocr的image_to_text()方法，传入image对象完成识别
# result = tesserocr.image_to_text(image)
print(verify_code)