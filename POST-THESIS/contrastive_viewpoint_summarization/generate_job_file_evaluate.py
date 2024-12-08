with open("jobs_evaluate_stability.txt", "w") as f:
    for year in ['06', '07', '08', '09', '10', '11', '12']:
        for category in ["political_parties", "political_ideologies_positions", "sects", "ethnicities", "countries_locations"]:
            f.write(f"--year {year} --category {category}\n")
f.close()


