years = list(range(1988, 2001))
archives = ['hayat']

with open('jobs_hayat.txt', 'w') as f:
    for ar in archives:
        for y in years:
            f.write(f"--archive {ar} --year {y}\n")