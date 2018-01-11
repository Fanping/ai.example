import random
import string
import sys
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

font_path = '/Library/Fonts/Arial.ttf'
number = 4
size = (60, 30)
bgcolor = (255, 255, 255)
fontcolor = (0, 0, 255)
linecolor = (255, 0, 0)
draw_line = True
line_number = (1, 5)


def gene_text():
    source = list(string.ascii_letters)
    for index in range(0, 10):
        source.append(str(index))
    return ''.join(random.sample(source, number))


def gene_line(draw, width, height):
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill=linecolor)


def gene_code(type, id):
    width, height = size
    image = Image.new('RGBA', (width, height), bgcolor)
    font = ImageFont.truetype(font_path, 25)
    draw = ImageDraw.Draw(image)
    text = gene_text()
    font_width, font_height = font.getsize(text)
    draw.text(((width - font_width) / number, (height - font_height) / number),
              text, \
              font=font, fill=fontcolor)
    if draw_line:
        gene_line(draw, width, height)
    image = image.transform((width + 20, height + 10), Image.AFFINE,
                            (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    image.save('../data/identification_codes/' + type + '/' + str(
        id) + '_' + text + '.png')


if __name__ == "__main__":
    for idx in range(10000):
        gene_code('train', idx)
    for idx in range(200):
        gene_code('text', idx)
