import img2pdf
import os
import argparse


def mkdir(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--archive', type=str, help="name of the archive to process images for")
	parser.add_argument('--path', type=str, help="path to the batch of images to convert to pdf")
	args = parser.parse_args()

	if not os.path.exists(args.path):
		raise ValueError(f"{args.path} is not a valid path / does not exist")

	# This is saved on the Local Disk of the Bliss 132 lab ONLY
	archives2outdirs = {
		"nahar": "images/nahar-images-pdf/",
		"assafir": "images/assafir-images-pdf/",
		"hayat": "images/hayat-images-pdf/"
	}

	all_paths = []
	for file in os.listdir(args.path):
		if file.startswith("._"):
			continue
		if file.endswith(".hocr"):
			continue
		if not file[0].isdigit():
			continue

		if ".tif" in file or ".TIF" in file:
			mkdir(archives2outdirs[args.archive])
			all_paths.append(os.path.join(args.path, file))

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
