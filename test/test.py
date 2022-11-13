import pywebio
from pywebio import *
from pywebio.input import *
from pywebio.output import *
from PIL import Image
# import cv2
import numpy as np
import io
from io import BytesIO
import matplotlib.pyplot as plt
# import plotly
# import plotly.graph_objects as go

# https://stackoverflow.com/questions/49511753/python-byte-image-to-numpy-array-using-opencv

def conv_1(image_bytes):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return decoded
def conv_2(image_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    return image
def png_bytes_to_numpy(png):
    """Convert png bytes to numpy array
    https://gist.github.com/eric-czech/fea266e546efac0e704d99837a52b35f

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))

def to_np(im):
    im.load()
    e = Image._getencoder(im.mode, 'raw', im.mode)
    e.setimage(im.im)

    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))

    mem = data.data.cast('B', (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("Encoder error %d in tobytes" %s)
    return data


def main():
    with use_scope('image'):
        get_img = file_upload("Select some pictures:", accept="image/*", multiple=False)
        imc = get_img['content']
        # im = Image.frombytes('RGB', (100,100) ,im)

        im = png_bytes_to_numpy(imc)
        print(type(im))
        put_code(im)
        # put_image(im)
        # im.resize((120,120))
        # n = to_np(im)
        # put_code(n)


        # n = Image.fromarray(np.uint8(im), 'RGB')
        # put_image(n)


        pil_img = Image.fromarray(im).convert('RGB')
        print(pil_img.mode)
        print(type(pil_img))
        # RGB

        pil_img.save('test.jpg')

        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        print(type(img_bytes))
        put_image(img_bytes)
        # https://zenn.dev/tamanobi/articles/88dacd450f8405c9a5a9

def main2():

    with use_scope('scope1'):  # open and enter a new output: 'scope1'
        put_text("placeholder")  # output text to scope1

        imgs = file_upload("Select some pictures:", accept="image/*", multiple=True)
        ary_lst = []
        ary_lst2 = []
        chk_lst = []
        for img in imgs:
            put_image(img['content'])
            as_ary = conv_1(img['content'])
            as_ary2 = conv_2(img['content'])
            ary_lst.append(as_ary)
            ary_lst2.append(as_ary2)
            chk = as_ary == as_ary2
            chk_lst.append(chk)
        put_code(ary_lst)
        put_text(len(ary_lst), chk_lst)


        imgar = Image.fromarray(ary_lst2[0], 'RGB')
        put_image(imgar)


if __name__ == '__main__':
    start_server(main, port=8888, debug=True)
