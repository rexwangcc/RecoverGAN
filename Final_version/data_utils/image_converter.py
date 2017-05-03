# Image Converter
# Rex Wang
import os
from time import time
from tqdm import trange
from tqdm import tqdm
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import freeze_support


class Converter(object):

    """docstring for Converter"""

    def __init__(self, work_directory):
        self.work_directory = work_directory
        self.filelist = []

    def check_and_convert_Grey(self, filename):
        img = Image.open(filename)
        rgbimg = Image.new("RGBA", img.size)
        rgbimg.paste(img)
        rgbimg.save('t' + filename)

    def excute(self):
        os.chdir(self.work_directory)
        for file in tqdm(os.listdir(self.work_directory)):
            self.check_and_convert_Grey(file)
            self.filelist.append(file)
        for file in tqdm(self.filelist):
            os.remove(file)

    def rename(self, extension='.jpg'):
        os.chdir(self.work_directory)
        counter = 1
        for file in tqdm(os.listdir(self.work_directory)):
            os.rename(file, str(counter) + extension)
            counter += 1

if __name__ == '__main__':
    # task1 = Converter(
    #     work_directory='')
    # task1.excute()
    # task1.rename()
