import sys
sys.path.append('E:\\repo_Deep-Learning-Thesis\\Multimodal_All_3_SDK')

import h5py
import time
import requests
from tqdm import tqdm 
import os
import math
import sys
from mmsdk.mmdatasdk import log


def read_URL(url,destination):
	if destination is None:
		log.error("Destination is not specified when downloading data",error=True)

	# print(destination)
	# #destination = "cmumosei_raw;/http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWords.csd"
	# print(destination.rsplit(os.sep,1))
	# temp = [1,2,3]
	# print(os.path.isdir("E:\repo_Deep-Learning-Thesis\Multimodal_All_3_SDK"))

	# with open("./hi.csd", 'w') as f:
	# 		f.write("haha")		
	# quit()

	# if os.path.isdir(destination.rsplit(os.sep,1)[-2]) is False:
	# 	os.mkdir(destination.rsplit(os.sep,1)[-2])

	# if(os.path.isfile(destination)):
	# 	log.error("%s file already exists ..."%destination,error=True)

	r = requests.get(url, stream=True)
	if r.status_code != 200:
		log.error('URL: %s does not exist'%url,error=True) 
	# Total size in bytes.
	total_size = int(r.headers.get('content-length', 0)); 
	block_size = 1024
	unit=total_size/block_size
	wrote = 0 
	with open(destination, 'wb') as f:
		log.status("Downloading from %s to %s..."%(url,destination))
		pbar=log.progress_bar(total=math.ceil(total_size//block_size),data=r.iter_content(block_size),postfix="Total in kBs",unit='kB', leave=False)
		for data in pbar:#unit_scale=True,
			wrote = wrote  + len(data)
			f.write(data)
	pbar.close()

	if total_size != 0 and wrote != total_size:
		log.error("Error downloading the data to %s ..."%destination,error=True)

	log.success("Download complete!")
	return True

if __name__=="__main__":
	read_URL("http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd", "./hi.csd")
