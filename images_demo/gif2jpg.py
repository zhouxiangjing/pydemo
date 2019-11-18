from PIL import Image

#将gif图片转成PNG图片
im = Image.open('C:/Users/86529/Desktop/VerifyCode.gif')


def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0:
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass


for i, frame in enumerate(iter_frames(im)):
    frame.save('C:/Users/86529/Desktop/VerifyCode.jpg', **frame.info)