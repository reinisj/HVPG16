# Prediction of HVPG in patients with liver cirrhosis

Code repository for the manuscript **Machine learning models for the assessment of portal hypertension based on standard laboratory tests in patients with compensated cirrhosis** (currently in submission).

<p align="left"><img src="HVPG16_graphical_abstract.svg " width="600"/></p>

**Hepatic venous pressure gradient (HVPG)** is an important prognosis marker in advanced chronic liver disease (cirrhosis). An HVPG value equal to or above 16 mmHg indicates high probability of serious complications (decompensation). Because HVPG is measured by an invasive procedure, its prediction from other clinical parameters is highly desirable.

In this project, we first explored data collected in the [Vienna Cirrhosis Study](https://clinicaltrials.gov/ct2/show/NCT03267615) for compensated patients (without apparent serious symptoms). Each patient record contained an HVPG measurement as well as a large number of other clinical parameters. The aim was to select 3 or 5 optimal parameters for **detection of high-risk patients** with an HVPG value equal to or above 16. The following markers were selected by recursive feature elimination:
**PLT** (platelets), **BILI** (bilirubin),  **aPTT** (activated partial thromboplastin time), **CHE** (cholinesterase), **GGT** (gamma-glutamyl transpeptidase).

The performance of 5 machine learning models predicting HVPG >= 16 was compared, measured by mean AUC score from repeated cross-validation. The top-performing model was logistic regression using the 5 selected parameters, reaching a clinically relevant AUC score of 0.878.

In the second part of the project, we validated our model on a large set of 840 patients from 7 clinical institutions. The best performing models reached an AUC score of 0.811 (5 parameters used) and 0.745 (PLT, BILI, aPTT).

As a companion to the paper, we have also developed an online HVPG predictor, available here:
[online calculator](https://liver.at/vlsg/hvpg16/)
