import cv2
import os
import glob


bg_data_dir = '/home/bowen/bg/'
bg_path = os.listdir(bg_data_dir)
for filename in bg_path:
	line  = 'bg/' + filename + '\n'
	#print(line)
	with open('/home/bowen/bg.txt', 'a') as f:
		f.write(line)
print('{} of negative samples written in bg.txt'.format(len(bg_path)))


pos_data_dir = '/home/bowen/14/'
img_path = os.listdir(pos_data_dir)
for i in img_path:
	image = cv2.imread(pos_data_dir + i)
	h = image.shape[0]
	w = image.shape[1]
	line = '14/' + i + ' 1 ' + '0 0 {} {}'.format(w,h) + '\n'
	with open('/home/bowen/info.dat', 'a') as f:
		f.write(line)

print('{} of positive samples written in info.dat'.format(len(img_path)))
