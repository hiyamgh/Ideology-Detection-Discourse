nahar_dir1 = 'nahar-images/nahar-batch-1/'
nahar_dir2 = 'nahar-images/nahar-batch-2/'
nahar_dir3 = 'nahar-images/nahar-batch-3/'
nahar_dir4 = 'nahar-images/nahar-batch-4/'

# assafir directories
assafir_dir1 = 'assafir-images/assafir-batch-1/'
assafir_dir2 = 'assafir-images/assafir-batch-2/'

# hayat directories
hayat_dir1 = 'hayat-images/hayat-batch-1/'
hayat_dir2 = 'hayat-images/hayat-batch-2/'

all_dirs = [
    nahar_dir1,
    nahar_dir2,
    nahar_dir3,
    nahar_dir4,

    assafir_dir1,
    assafir_dir2,

    hayat_dir1,
    hayat_dir2
]

with open('jobs.txt', 'w') as f:
    for dirname in all_dirs:
        if "nahar" in dirname:
            f.write(f"--archive nahar --path {dirname}\n")
        elif "assafir" in dirname:
            f.write(f"--archive assafir --path {dirname}\n")
        else:
            f.write(f"--archive hayat --path {dirname}\n")