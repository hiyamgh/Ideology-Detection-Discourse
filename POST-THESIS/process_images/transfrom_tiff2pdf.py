import img2pdf
import os

def mkdir(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)


if __name__ == '__main__':
	nahar_dir1 = 'F:/newspapers/nahar-images/nahar-batch-1/'
	nahar_dir2 = 'F:/newspapers/nahar-images//nahar-batch-2/'
	nahar_dir3 = 'F:/newspapers/nahar-images/nahar-batch-3/'
	nahar_dir4 = 'F:/newspapers/nahar-images/nahar-batch-4/'

	# assafir directories
	assafir_dir1 = 'F:/newspapers/assafir-images/assafir-batch-1/'
	assafir_dir2 = 'F:/newspapers/assafir-images/assafir-batch-2/'

	# hayat directories
	hayat_dir1 = 'F:/newspapers/hayat-images/hayat-batch-1/'
	hayat_dir2 = 'F:/newspapers/hayat-images/hayat-batch-2/'

	nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3, nahar_dir4]  # all nahar directories in one list - helps in looping
	assafir_dirs = [assafir_dir1, assafir_dir2]  # all nahar directories in one list - helps in looping
	hayat_dirs = [hayat_dir1, hayat_dir2]  # all nahar directories in one list - helps in looping

	archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs, "hayat": hayat_dirs}

	# This is saved on the Local Disk of the Bliss 132 lab ONLY
	archives2outdirs = {"nahar": "E:/POST-THESIS/nahar-images-pdf/",
						"assafir": "E:/POST-THESIS/assafir-images-pdf/",
						"hayat": "E:/POST-THESIS/hayat-images-pdf/"}

	all_paths = []
	for archive in archive2dirs:
		for dir in archive2dirs[archive]:
			rootdir = dir
			for subdir, dirs, files in os.walk(rootdir):
				print(subdir)
				for file in files:
					if file.startswith("._"):
						continue
					if file.endswith(".hocr"):
						continue

					if not file[0].isdigit():
						continue

					if ".tif" in file or ".TIF" in file:
						mkdir(archives2outdirs[archive])
						all_paths.append(os.path.join(subdir, file))

	for path in all_paths:
		filename = path.split("/")[-1]
		if "nahar" in path:
			try:
				with open(os.path.join(archives2outdirs["nahar"], "{}.pdf".format(filename[:-4])), "wb") as f:
					f.write(img2pdf.convert(path))
			except:
				print(f"problem with {path}")

		elif "assafir" in path:
			try:
				with open(os.path.join(archives2outdirs["assafir"], "{}.pdf".format(filename[:-4])), "wb") as f:
					f.write(img2pdf.convert(path))
			except:
				print(f"problem with {path}")

		else:
			try:
				with open(os.path.join(archives2outdirs["hayat"], "{}.pdf".format(filename[:-4])), "wb") as f:
					f.write(img2pdf.convert(path))
			except:
				print(f"problem with {path}")




	# # convert all files ending in .jpg inside a directory
	# 	dirname = "input/"
	# 	imgs = []
	# 	for fname in os.listdir(dirname):
	# 		if not fname.endswith(".tif"):
	# 			continue
	# 		path = os.path.join(dirname, fname)
	# 		if os.path.isdir(path):
	# 			continue
	# 		imgs.append(path)
	#
	# 	count = 0
	# 	for path in imgs:
	# 		with open(os.path.join("output/", "{}.pdf".format(count)), "wb") as f:
	# 			f.write(img2pdf.convert(path))
	# 			count += 1
