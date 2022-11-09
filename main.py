from PIL import Image
import cv2
import numpy as np
import timeit
from matplotlib import pyplot as plt


im = Image.open('./files/manu.jpg').resize((120, 120))
asnumpy = np.asarray(im)


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

n = to_np(im)

p = np.all(n==asnumpy)
grey = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)


img = Image.fromarray(grey, 'L') #'RGB', 'L'
# img = Image.fromarray(n, 'RGB') #'RGB', 'L'
img.save('files/my.png')
img.show()

# plt.imshow(grey, interpolation='nearest')
# print(plt.show())