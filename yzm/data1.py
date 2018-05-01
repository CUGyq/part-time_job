from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 验证码中的字符, 就不用汉字了
CAPTCHA_IMAGE_PATH = 'F:/python/yzmreg/train/'
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image(captchaImgPath=CAPTCHA_IMAGE_PATH):
    while(1):
        image = ImageCaptcha()

        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)


        captcha = image.generate(captcha_text)
        image.write(captcha_text, captchaImgPath + captcha_text + '.jpg')

        captcha_image = Image.open(captcha)
        #captcha_image.show()
        captcha_image = np.array(captcha_image)
        if captcha_image.shape==(60,160,3):
            break








if __name__ == '__main__':

    gen_captcha_text_and_image(CAPTCHA_IMAGE_PATH)




