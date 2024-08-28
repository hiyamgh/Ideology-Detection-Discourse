years = list(range(1982, 2012))
archives = ['assafir']

with open('jobs_assafir.txt', 'w') as f:
    for ar in archives:
        for y in years:
            f.write(f"--archive {ar} --year {y}\n")