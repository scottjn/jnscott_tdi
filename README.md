# Targeted Training: Improving healthcare outcomes and cutting Medicare expenditures using data science
## J. Nathan Scott
### This is my capstone project for The Data Incubator, in partial fulfillment of the requirements of their Fellowship program.
updated: June 6, 2019

## Motivation

Healthcare is big business. According to the Centers for Medicare and Medicaid Services (CMMS), healthcare spending in the U.S. grew 3.9 percent in 2017, reaching $3.5 trillion or $10,739 per person. This is equivalent to 17.9% of the nation's GDP. Moreover, it is projected to grow at an average annual rate of 5.5% per year for 2018-2027. If that projection bears out, annual healthcare spending in the U.S. will reach **_$6.0 trillion_** by 2027. This is 0.8% faster than projected growth in GDP over the same period. Medicare enrollment in particular will drive that increase.

These sums, staggering though they may be, also hint at the potential for vast savings. The CMMS administers a program that scores medical providers on the quality of care they give with respect to many 
criteria. For instance, providers may be scored on how well they do, as evaluated by patients, when it comes to recommending the patient receive a flu vaccine. Flu is a serious illness, particularly in 
the elderly, sometimes resulting in hospitilization and death. Another scoring example is smoking cessation. Physicians can receive a score based on their discussing tobacco use or cessation with their 
patients. Smoking is still a leading cause of death amd a tremendous healthcare burden and expense in the U.S. If we assume that medical providers have some efficacy in their treatment of patients, and 
that there is some room for improvement, even if only marginal, in how well those providers treat their patients, then the potential for savings is tremendous. A clear role for private business could be 
in targeting training/education materials to medical provider populations where improvements in scores, and therefore in efficacy, will translate into improved health outcomes and reduced healthcare 
expenditures.

This goal of this project is to quantify and discover underlying trends in data concerning the quality of care rendered by medical professionals to their patients.

Vocabulary:
Measures: These are the specific areas or facets of care that patients score providers on.

Specifically, the objectives are as follows:

1) Identify measures which:
 - Affect a large number of patients
 - Involve a large number of practitioners
 - Suffer from particularly low scores 
 - Have 

2) For measures identified, use machine learning tecniques to identify features of greatest importance in determining a provider's score.

The data used in this project is all publicly accessible and freely available. The locations and basic descriptions of the files utilized are below.

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

us_census_bureau_regions_and_divisions.csv
1.7 kB
51 x 4

https://data.medicare.gov/data/hospital-compare
HCAHPS - Hospital.csv --> hcahps_hospital.csv

101 MB

All results stated are based on versions of datasets downloaded on May 30, 2019. In brief, the iep2916 csv file contains records related to the score of a given provider for a given measure. The ndf2016 
csv file cotains records for medical providers with information such as the year they graduated with their highest degree, where they went to medical school, which medical practice(s) they are members 
of, what hospitals they practice at, etc. The us_census_bureau_regions_and_divisions csv file bins states into geographical regions and divisions. The hcahps_hospital csv file contains scoes similar to 
those being analyzed for the medical providers but instead for hospitals.
