# Image Data Modifier
# Rex Wang
import os
from time import time
from tqdm import trange
from tqdm import tqdm
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import freeze_support


class Modifier(object):
    """docstring for Modifier"""

    def __init__(self, work_directory, counter_num):
        self.work_directory = work_directory
        self.counter_num = counter_num
        self.to_remove_set = set()

    def check_FAIL(self, filename):
        name_to_check = filename + '_FAIL'
        if os.path.exists(self.work_directory + name_to_check):
            self.to_remove_set.add(self.work_directory + name_to_check)
            self.to_remove_set.add(self.work_directory + filename + '.jpg')
            # self.to_remove_set.add(self.work_directory + filename)

    def remove_INVALID(self, filename_with_path):
        os.remove(filename_with_path)

    def excute(self):
        for i in trange(1, self.counter_num + 1):
            self.check_FAIL(filename=str(i))

        for file in tqdm(self.to_remove_set):
            try:
                self.remove_INVALID(file)
            except:
                pass

if __name__ == '__main__':
    # task1 = Modifier(
    #     work_directory='', counter_num=17556)
    # task1.excute()
