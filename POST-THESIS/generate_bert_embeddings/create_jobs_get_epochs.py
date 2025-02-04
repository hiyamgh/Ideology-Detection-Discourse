with open("jobs_epochs.txt", "w") as f:
    for archive in ["An-Nahar", "As-Safir"]:
        for year in ['1983', '1984', '1989', '1990', '1991', '1992', '1993', '1994', '1995']:
            f.write(f"--archive_name {archive} --year {year}\n")
    f.close()