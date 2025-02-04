## Step 1 for teh year 1982
* run `get_opinionated.py`: will join only opinionated articles together in one directory saved into `opinionated_articles/1982/txt_files/An-Nahar` + `opinionated_articles/1982/txt_files/As-safir`
* run `get_mlm_no_trainer.py`
*

## Step 2 for the rest of the years (EPOCHS):
* run `create_jobs_get_epochs.py` to create a jobs (.txt) file `jobs_epochs.txt` (one job per line) in the same directory
* run `get_epochs.sub` which calls `get_epochs.py` that uses the jobs created in the bullet above (`jobs_epochs.txt`). This will collect training (.txt) files, one per archive per year, saved into `opinionated_articles_DrNabil/{year}/txt_files/{archive_name}/` and `opinionated_articles_DrNabil/{year}/training_file/{archive_name}/`
  * `opinionated_articles_DrNabil/{year}/txt_files/{archive_name}/`: contains all teh training (.txt) files under a specified `year` and `archive_name`
  * `opinionated_articles_DrNabil/{year}/training_file/{archive_name}/`: is the compilation of all training files in the bullet above into one training under a specified `year` and `archive_name`
* run `get_epochs.sub` which calls `get_epochs.py` with jobs inside `jobs_epochs.txt` (which are created using `create_jobs_get_epochs.py`)
* run `run_mlm_no_trainer.sub` which calls `run_mlm_no_trainer.py` with jobs inside `jobs_mlm_epochs.txt` for fine tuning on the `yearly An-Nahar + As-Safir` training (.txt) files through `AutoModelForCausalML`. This will save fine tuned models inside the directory `trained_models/{archive_name}/{year}/{model_name}`
