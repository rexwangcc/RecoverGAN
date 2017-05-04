from PIL import Image
import glob
import os


class Resizer(object):

    def __init__(self,  filepath, extension='.jpg', resize_tuple=(64, 64), overwrite=True, outputpath=None):
        self.extension = extension
        assert(isinstance(resize_tuple, tuple))
        self.resize_tuple = resize_tuple
        self.filepath = filepath
        self.filelist = glob.glob(self.filepath + "*" + self.extension)
        self.overwrite = overwrite

        if not self.overwrite:
            assert(outputpath)
            if not os.path.exists(str(outputpath)):
                os.makedirs(str(outputpath))
            self.outputpath = outputpath

    def inplace_resize(self, file):
        img = Image.open(file)
        new_img = img.resize(self.resize_tuple, Image.ANTIALIAS)
        new_img.save(file)

    def outplace_resize(self, file):
        name = file.rsplit('/', 1)[-1]
        img = Image.open(file)
        new_img = img.resize(self.resize_tuple, Image.ANTIALIAS)
        new_img.save(os.path.join(self.outputpath, name))

    def run(self):
        if self.overwrite:
            map(self.inplace_resize, self.filelist)
        else:
            map(self.outplace_resize, self.filelist)


if __name__ == '__main__':
    # resize_task = Resizer(extension='.jpg', resize_tuple=(64, 64), overwrite=True, filepath='/')
    # resize_task.run()
