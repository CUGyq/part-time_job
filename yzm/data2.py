import sys
import os
import shutil
import random
import time
# captcha是用于生成验证码图片的库，可以 pip install captcha 来安装它
from captcha.image import ImageCaptcha
class DATA:
    def __init__(self,trainPath,testPath,testNum):
        self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.alphabet  = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z']
        self.ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z']

        self.CHAR_SET = self.alphabet + self.ALPHABET
        self.CHAR_SET_LEN = len(self.CHAR_SET)
        self.CAPTCHA_LEN = 4

        self.CAPTCHA_IMAGE_PATH = trainPath
        self.TEST_IMAGE_PATH = testPath
        self.TEST_IMAGE_NUMBER = testNum
    def get_image(self):
        k = 0
        total = 1
        for i in range(self.CAPTCHA_LEN):
            total *= self.CHAR_SET_LEN
        for i in range(self.CHAR_SET_LEN):
            for j in range(self.CHAR_SET_LEN):
                for m in range(self.CHAR_SET_LEN):
                    for n in range(self.CHAR_SET_LEN):
                        captcha_text = self.CHAR_SET[i] + self.CHAR_SET[j] + self.CHAR_SET[m] + self.CHAR_SET[n]
                        image = ImageCaptcha()
                        image.write(captcha_text, self.CAPTCHA_IMAGE_PATH + captcha_text + '.jpg')
                        k += 1
                        sys.stdout.write("\rCreating %d/%d" % (k, total))
                        sys.stdout.flush()
    def get_test(self):
        fileNameList = []
        for filePath in os.listdir(self.CAPTCHA_IMAGE_PATH):
            captcha_name = filePath.split('/')[-1]
            fileNameList.append(captcha_name)
        random.seed(time.time())
        random.shuffle(fileNameList)
        for i in range(self.TEST_IMAGE_PATH):
            name = fileNameList[i]
            shutil.move(self.CAPTCHA_IMAGE_PATH + name, self.TEST_IMAGE_PATH + name)

if __name__ == '__main__':
    CAPTCHA_IMAGE_PATH = 'F:/python/yzmreg/train/'
    # 用于模型测试的验证码图片的存放路径，它里面的验证码图片作为测试集
    TEST_IMAGE_PATH = 'F:/python/yzmreg/test/'
    data = DATA(CAPTCHA_IMAGE_PATH,TEST_IMAGE_PATH,50)
    data.get_image()
    data.get_test()
