import img2pdf
import os
import argparse


def mkdir(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--archive', type=str, help="name of the archive to process images for")
	parser.add_argument('--year', type=str, help="the year for which to consider converting images to pdfs")
	args = parser.parse_args()

	nahar_dir1 = 'nahar-images/nahar-images/nahar-batch-1/'
	nahar_dir2 = 'nahar-images/nahar-images/nahar-batch-2/'
	nahar_dir3 = 'nahar-images/nahar-images/nahar-batch-3/'
	nahar_dir4 = 'nahar-images/nahar-images/nahar-batch-4/'
	nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3, nahar_dir4]

	# assafir directories
	assafir_dir1 = 'assafir-images/assafir-images/assafir-batch-1/'
	assafir_dir2 = 'assafir-iamges/assafir-images/assafir-batch-2/'
	assafir_dirs = [assafir_dir1, assafir_dir2]

	# hayat directories
	hayat_dir1 = 'hayat-images/hayat-images/hayat-batch-1/'
	hayat_dir2 = 'hayat-images/hayat-images/hayat-batch-2/'
	hayat_dirs = [hayat_dir1, hayat_dir2]

	archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs, "hayat": hayat_dirs}

	# This is saved on the Local Disk of the Bliss 132 lab ONLY
	archives2outdirs = {
		"nahar": "nahar-images-pdf/{}/".format(args.year),
		"assafir": "assafir-images-pdf/{}/".format(args.year),
		"hayat": "hayat-images-pdf/{}/".format(args.year)
	}

	all_paths = []
	for dir in archive2dirs[args.archive]:
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
				if file[:2] == args.year[2:4]:
					if ".tif" in file or ".TIF" in file or ".jpg" in file:
						mkdir(archives2outdirs[args.archive])
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
