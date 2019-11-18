from PIL import Image


def to_jpg(src_img, dst_img):
    try:
        img = Image.open(src_img)
        img = img.convert('RGB')
        img.save(dst_img, quality=95)
        return True
    except IOError:
        return False


def to_png(src_img, dst_img):
    try:
        img = Image.open(src_img)
        img.save(dst_img)
        return True
    except IOError:
        return False
    pass


def gif_iter_frames(im):
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


def gif_to_png(src_img, dst_img):
    try:
        img = Image.open(src_img)
        for i, frame in enumerate(gif_iter_frames(img)):
            frame.save(dst_img, **frame.info)
        return True
    except IOError:
        return False
    pass