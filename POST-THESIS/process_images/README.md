## Assafir - Image Statistics
Count all file extensions recursively in a folder, and display them in a sorted fashion:
```commandline
 cd assafir-images
 find . -type f -name '*.*' -not -name '.*' | sed -Ee 's,.*/.+\.([^/]+)$,\1,' | sort | uniq -ci | sort -n
```
outputs:
```
     40 jpg
    134 TIF
 184973 tif
```

## Hayat - Image Statistics
Count all file extensions recursively in a folder, and display them in a sorted fashion:
```commandline
 cd assafir-images
 find . -type f -name '*.*' -not -name '.*' | sed -Ee 's,.*/.+\.([^/]+)$,\1,' | sort | uniq -ci | sort -n
 145460 TIF
```
outputs:
```
 145460 TIF
```


## Nahar - Image Statistics
Count all file extensions recursively in a folder, and display them in a sorted fashion:

```commandline
 cd assafir-images
 find . -type f -name '*.*' -not -name '.*' | sed -Ee 's,.*/.+\.([^/]+)$,\1,' | sort | uniq -ci | sort -n
 145460 TIF
```
outputs:
```
      1 sh
      2 jpg
  80783 TIF
 197866 tif
```
totaling 278,651 whilst the number reported in Jad's paper is 278,779 --> 128 images missing


## Steps to transform images to PDF files:
1. Upload the image zip files (.zip) to the cluster
2. Fix the uploaded zip files using:
   * ``fix_nahar.sub``
   * ``fix_hayat.sub``
   * ``fix_assafir.sub``
3. Extract the repaired files by running:
    * ``extract_nahar.sub``
    * ``extract_assafir.sub``
    * ``extract_hayat.sub``
4. Create job files (for batch arrays) by running:
    * ``create_jobs_file_assafir.py``
    * ``create_jobs_file_nahar.py``
    * ``create_jobs_file_hayat.py``
5. Run the script that transforms, by year, each image into ``.pdf``:
   * ``transform_assafir.sub``
   * ``transform_nahar.sub``
   * ``transform_hayat.sub``
6. The name of the folders (one folder per year in the ``nahar-images-pdf``, ``assafir-images-pdf``, and ``hayat-images-pdf`` had ``'\r'`` char, so please run the following script after you `cd` into each directory)
   * ``rename_folders.py``
   

## Diagnostics
* Just check file names (some images contain the letter ``m`` so making sure if this is true)
* Run ``check_file_names.py`` using ``check_file_names.sub``