## `Original Data` Directory

Assumes that `nahar.tar.gz` and `assafir.tar.gz` exist

### Extract the .tar.gz zip files
1. `extract_assafir.sub`: extracts assafir .out files (taken from Jad) into the directory assafir/
2. `extract_nahar.sub`: extracts nahar .out files (taken from Jad) into the directory nahar/

### Split the data - monthly level
* Each file in the archive is named as follows: YYMMDDPP.txt:
    * YY: is the year, if 89 for example, then its 1989, if 05 for example, its 2005
    * MM: month number, from 01-12
    * DD: Day number, from 01-31
    * PP: page number, from 01 ...
* run `split_normalize_text.py`: This script does the following:
    * splits the newspaper files at the monthly level (i.e. all files belonging to the same year-month are collated into one .txt file)
    * For every sentence:
        * Arabic normalization is applied
        * Arabic stopwords are removed, if any
    * the monthly splits are saved in `txt_files/archive_name/monthly/year-month.txt`
