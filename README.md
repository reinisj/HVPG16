# Prediction of HVPG in patients with liver cirrhosis

Code repository for the manuscript **Machine learning models for the assessment of portal hypertension based on standard laboratory tests in patients with compensated cirrhosis** (currently in submission).

<p align="left"><img src="graphical_abstract.png " width="600"/></p>

**Hepatic venous pressure gradient (HVPG)** is an important prognosis marker in advanced chronic liver disease (cirrhosis). An HVPG value ≥ 10 mmHg is associated with a risk of serious complications (decompensation), HVPG ≥ 16 mmHg indicates a very high risk. Because it is measured by an invasive procedure, prediction of HVPG from other clinical parameters is highly desirable.

In this project, we first explored data collected in the [Vienna Cirrhosis Study](https://clinicaltrials.gov/ct2/show/NCT03267615) from compensated patients (without apparent serious symptoms). Each patient record contained an HVPG measurement as well as a large number of other clinical parameters. The aim was to select 3 or 5 optimal parameters for **detection of high-risk patients** with an HVPG value equal to or above 16. The following markers were chosen, based on recursive feature elimination:
 - **3P parameter set:** **PLT** (platelets), **BILI** (bilirubin),  **INR** (international normalized ratio)
 - **5P parameter set:** **PLT** (platelets), **BILI** (bilirubin),  **aPTT** (activated partial thromboplastin time), **GGT** (gamma-glutamyl transpeptidase), **CHE** (cholinesterase).

The performance of 5 machine learning models predicting HVPG ≥ 16 was compared, measured by mean AUC score from repeated cross-validation:
 - logistic regression
 - multilayer perceptron (MLP)
 - random forest
 - support vector machine (SVM)
 - eXtreme Gradient Boosting (XGBoost)

Here, the top-performing model was logistic regression in the 5P setting, reaching a clinically relevant AUC score of 0.887.

In the second part of the project, we validated our model on a large set of 1232 patients from 7 clinical institutions, now both in the HVPG ≥ 10 and HVPG ≥ 16 settings. The best performing model reached AUC scores of 0.773 in the HVPG10 setting (logistic regression, 3P) and 0.812 in HVP16 (logistic regression, 5P).

As a companion to the paper, we have also developed an online HVPG predictor, available here:
[online calculator](https://liver.at/vlsg/hvpg16/)
