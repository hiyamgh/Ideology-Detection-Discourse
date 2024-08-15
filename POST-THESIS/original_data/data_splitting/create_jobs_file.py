years = list(range(1982, 2012))
months = list(range(1, 13))
archives = ['nahar', 'assafir']

with open('jobs.txt', 'w') as f:
    for ar in archives:
        for y in years:
            if y > 2009 and ar == 'nahar':
                continue
            for m in months:
                f.write(f"--archive {ar} --month {m} --year {y}\n")