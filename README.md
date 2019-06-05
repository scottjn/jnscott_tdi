# jnscott_tdi
This goal of this project is to quantify and discover underlying trends in data concerning the quality of care rendered by medical professionals to their patients. This data is collected by the Centers 
for Medicare & Medicaid Services (CMS).

Vocabulary:
Measures: These are the specific areas or facets of care that patients score providers on.

The care rendered by medical providers is complex and multifaceted. Likewise, the actual impact of a given doctor's advice or instructions on a given patient's health outcome is impossible to calculate. 
However decades of research have shown that medcal providers can have significant impacts on patient's decisions

Specifically, the business objectives are as follows:

1) Identify measures which:
 - Affect a large number of patients
 - Involve a large number of practitioners
 - Suffer from particularly low scores 

2) For measures identified, use machine learning tecniques to identify features of greatest importance in determining a provider's score.
3) Identify provider primary specialties that could benefit from 

The data used in this project is all publicly accessible and freely available. The locations and basic descriptions of the files utilized in this project are below.

Information on the Physician Compare Initiative:
https://www.medicare.gov/physiciancompare/#about/aboutPhysicianCompare
https://www.medicare.gov/physiciancompare/#about/improvinghealthcarequality

"History of Physician Compare
The Centers for Medicare & Medicaid Services (CMS) created Physician Compare in December 2010 as required by the Affordable Care Act. We continually work to be sure it’s easy to use and includes the 
most useful information to help you find clinicians to meet your health care needs. The performance information on Physician Compare helps you make informed decisions about your health care and also 
encourages clinicians to provide you the best care."

Supporting Documentation/Data Dictionary:
https://data.medicare.gov/views/bg9k-emty/files/f3a6ce79-f204-46d5-802b-2c343c289764?filename=Physician_Compare_Data_Dictionary_2016.pdf&content_type=application%2Fpdf%3B%20charset%3Dbinary

2016 Data downloaded from:
https://data.medicare.gov/views/bg9k-emty/files/42c584ca-5ccc-4c51-a834-34e3ae35e109?content_type=application%2Fzip%3B%20charset%3Dbinary&filename=Physician_Compare.zip

Prior to processing, the following files from that archive were renamed as follows.

Physician_Compare_2016_Individual_EP_Public_Reporting.csv --> iep2016.csv
Physician_Compare_National_Downloadable_File.csv --> ndf2016.csv

iep2016.csv
37 MB
325263 x 11

ndf2016.csv
559 MB
2060816 x 41

Census Regional designnation data downloaded from:
https://github.com/cphalpert/census-regions/blob/master/us%20census%20bureau%20regions%20and%20divisions.csv

us_census_bureau_regions_and_divisions.dsv
1.7 kB
51 x 4

https://data.medicare.gov/data/hospital-compare
HCAHPS - Hospital.csv --> hcahps_hospital.csv

101 MB

Datasets were updated on May 16, 2019. All results stated are based on versions of datasets downloaded on May 30, 2019.
In brief, the iep2916 csv file contains records related to the score of a given provider for a given measure. The ndf2016
csv file cotains records for medical providers with information such as the eyar they graduated with their highest degree,
where they went to medical school, which medical practice(s) they are members of, what hospitals they practice at, atc.
The 






















