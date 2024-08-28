years = list(range(1982, 2010))
archives = ['nahar']

with open('jobs_nahar.txt', 'w') as f:
    for ar in archives:
        for y in years:
            f.write(f"--archive {ar} --year {y}\n")