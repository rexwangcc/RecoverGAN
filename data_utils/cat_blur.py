import os
import random
import datetime
import glob
import pip
# from PIL import Image
# from PIL import ImageFilter

class cat_face_blur(object):
	def __init__(self,blur_size = 0.7,Gaussian_blur = 20,path_to_blur = 'to_blur/',extension = '.jpg'):
		self.blur_size = blur_size
		self.Gaussian_blur = Gaussian_blur
		self.path_to_blur = path_to_blur
		self.extension = extension
		self.files = Catface_jpg = [f for f in os.listdir(os.getcwd()+'/' + self.path_to_blur) if f.endswith(self.extension)]
		self.dim = len(self.files)
		print('*********************************************')
		print('Total Images to blur: ' + str(self.dim))
		print('The blur will cover' +str(self.blur_size * 100) + '%'+'of the images')
		print('The Gaussian blur degree is ' + str(self.Gaussian_blur))
		print('*********************************************')


	def import_or_install_pillow(self):
	    try:
	        __import__('pillow')
	    except ImportError:
	        pip.main(['install', 'pillow'])
	    # from PIL import Image
	    # from PIL import ImageFilter

	def saveImg(self,img,filename):
	    dir =os.path.dirname(os.getcwd()+'/blur_output/')
	    if not os.path.exists(dir):
	        os.makedirs(dir)
	    img.save(dir+'/'+filename)

	def main(self):
		self.import_or_install_pillow()
		from PIL import Image
		from PIL import ImageFilter
		if not self.blur_size < 0 and not self.blur_size >1 and not self.Gaussian_blur<0 and not self.Gaussian_blur >100:
			for i in range(self.dim):
				cat = Image.open(os.getcwd()+'/'+self.path_to_blur+self.files[i])
				width,height = cat.size
				w = int(width * self.blur_size)
				h = int(height * self.blur_size)
				range_w = width - w
				range_h = height - h
				seed = datetime.datetime.now().second
				random.seed(seed)
				start_w = random.randrange(range_w)
				start_h = random.randrange(range_h)
				box = (start_w,start_h,start_w + w,start_h + h)
				crop_cat = cat.crop(box)
				#             saveImg(crop_cat,Catface_jpg[i][:-4]+'_partial.jpg',output_batch)
				crop_cat = crop_cat.filter(ImageFilter.GaussianBlur(self.Gaussian_blur))
				cat.paste(crop_cat,box)
				self.saveImg(cat,self.files[i],)
		print('Blur Done!')
		print('Blur images are saved in folder "blur_output" ')


app = cat_face_blur()
app.main()
