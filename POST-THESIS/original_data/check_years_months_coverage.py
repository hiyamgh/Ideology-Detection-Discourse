import os


if __name__ == '__main__':
    nahar_dir1 = 'nahar/nahar/nahar-batch-1/'
    nahar_dir2 = 'nahar/nahar/nahar-batch-2/'
    nahar_dir3 = 'nahar/nahar/nahar-batch-3/'
    nahar_dir4 = 'nahar/nahar/nahar-batch-4/'

    # assafir directories
    assafir_dir1 = 'assafir/assafir/assafir-batch-1/'
    assafir_dir2 = 'assafir/assafir/assafir-batch-2/'

    nahar_dirs = [nahar_dir1, nahar_dir2, nahar_dir3, nahar_dir4]  # all nahar directories in one list - helps in looping
    assafir_dirs = [assafir_dir1, assafir_dir2]  # all nahar directories in one list - helps in looping

    archive2dirs = {"nahar": nahar_dirs, "assafir": assafir_dirs}

    years_months_nahar = []
    years_months_assafir = []

    years = list(range(1982, 2012))
    months = list(range(1, 13))
    archives = ['nahar', 'assafir']

    for ar in archives:
        for y in years:
            if y > 2009 and ar == 'nahar':
                continue
            for m in months:
                if ar == "nahar":
                    years_months_nahar.append(f"{y}-{m}")
                else:
                    years_months_assafir.append(f"{y}-{m}")

    found_years_months_nahar = set()
    found_years_months_assafir = set()
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
                    if ".txt" not in file:
                        continue
                    if not file[:2].isdigit():
                        continue

                    yearf = file[:2]
                    monthf = file[2:4]
                    dayf = file[4:6]
                    page_nbf = file[6:8]

                    if yearf[0] == '0':
                        year = int('200' + yearf[1])
                    elif int(yearf) <= 11:
                        year = int('20' + yearf[1])
                    else:
                        # print(yearf, file)
                        year = int('19' + yearf)

                    if monthf[0] == '0':
                        month = int(monthf[1])
                    else:
                        month = int(monthf)

                    if archive == 'nahar':
                        found_years_months_nahar.add(f"{str(year)}-{str(month)}")
                    else:
                        found_years_months_assafir.add(f"{str(year)}-{str(month)}")

    print(f"Year-months not in nahar: {sorted(set(years_months_nahar) - found_years_months_nahar)}")
    print(f"Year-months not in assafir: {sorted(set(years_months_assafir) - found_years_months_assafir)}")