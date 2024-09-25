## `Original Data` Directory

Assumes that `nahar.tar.gz` and `assafir.tar.gz` exist

### Extract the .tar.gz zip files
1. `extract_assafir.sub`: extracts assafir .out files (taken from Jad) into the directory assafir/
2. `extract_nahar.sub`: extracts nahar .out files (taken from Jad) into the directory nahar/

### Transform the data
* Each newspaper issue is composed of multiple pages, and each page is a `.txt` file in our corpora.
* For each `.txt` file, we collate all lines belonging to the same sentence, together
* We observe that a sentence spans multiple lines and there is, at least, one `'\n'` char between one sentence and the other
* Example:
````
ويقول. في اشارة الى الشائعات حول تسميم الزعيم الفلسطيني
«ستزداد الامور سوءا لأن الفلسطينيين فقدوا زعيمهم
وسيتهموننا الان اننا مسؤولين عن موته».

وتقول طبيبة نفسية انها «لن تبكي» على وفاة عرفات؛
وتضيف «كان يلغ من العمر 758 عاما من الطبيعي ان يموت
المرء من المرض عندما تحين ساعته». وتابعت «انه بالطيع ليس
شيطانا كما نتصوره في اسرائيل لكنه لم يتعامل معنا كما لو
كان ملاكا».
````
* run `transform_data.py` (found in `data_transformation/` directory) to do this, and each transformed `.txt` file is saved in the output directory: `archive_name_transformed/`

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
