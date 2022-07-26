# Jiří Reiniš
# jreinis@cemm.oeaw.ac.at

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import re

# prediction models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# metrics
from sklearn.metrics import plot_roc_curve, roc_auc_score, roc_curve, auc, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# do not show warnings
import warnings
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

# plotting configuration - set size, use high resolution
from IPython.display import set_matplotlib_formats
from matplotlib import rc_params
plt.rcParams['figure.figsize'] = [11, 5]
plt.rcParams['pdf.fonttype'] = 42
set_matplotlib_formats('retina')
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 11


def repeat_CV_get_AUC(models, names, n, X, y):
    """
    For each model, runs n 5-fold cross-validations on the provided data. Returns AUC scores from all CVs and fold for each model per fold, and statistics (mean/median AUC values + confidence intervals)

    Parameters
    ----------
    models : list
        List of sklearn models to compare
    names : list
        Names of the models to show in the reported statistics
    n : int
        Number of CVs to perform
    X : matrix/DataFrame
        Independent variables
    y : vector/DataFrame
        Dependent variable

    """    

    # used later so that each of the n cross validations is the same for each model
    random_states = np.random.randint(10000, size = n)
    
    # variable to store all AUCs
    AUCs = {names[i]:None for i in range(len(names))}
    
    for clf, name in zip(models, names):
        AUCs_model = []
        print(f"{name: <25}", end = "")
        for j in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = StratifiedKFold(n_splits=5, random_state = random_states[j], shuffle=True)
                AUCs_model.extend(get_AUC_CV(clf, cv, X, y))
                print("*", end = "")
        print()
        AUCs[name] = AUCs_model
                
    # calculate statistics to report
    mean_AUCs = [np.mean(x) for x in AUCs.values()]
    mean_AUCs_95CIs = [st.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=st.sem(x)) for x in AUCs.values()]
    median_AUCs = [np.median(x) for x in AUCs.values()]
    median_AUCs_95CIs = [st.t.interval(0.95, len(x)-1, loc=np.median(x), scale=st.sem(x)) for x in AUCs.values()]
    
    # create a dataframe out of these statistics
    scores_df = pd.DataFrame({"model": names,
                              "mean_AUC": mean_AUCs,
                              "mean_AUC_95_CI_range": np.array([x[1] for x in mean_AUCs_95CIs]) - np.array(mean_AUCs),
                              "median_AUC": median_AUCs,
                              "median_AUC_95_CI_range": np.array([x[1] for x in median_AUCs_95CIs]) - np.array(median_AUCs),
                             })
    return pd.DataFrame(AUCs), scores_df


# returns mean AUC
def get_AUC_CV(clf, cv, X, y):
    """
    For a model, runs n 5-fold cross-validations on the provided data. Returns AUC scores from each fold.

    Parameters
    ----------
    clf : sklearn model
        Classifier
    cv : StratifiedKFold object
        Specifies how to split the dataset into folds
    X : matrix/DataFrame
        Independent variables
    y : vector/DataFramew
        Dependent variable

    """  
    aucs = []  # save auc value for each fold (to calculate the mean)

    for i, (train, test) in enumerate(cv.split(X, y)):  
        clf.fit(X.iloc[train], y.iloc[train])
        y_probas = clf.predict_proba(X.iloc[test])[:,1]
        aucs.append(roc_auc_score(y.iloc[test], y_probas))    
    return aucs

 
def test_models_variables(data, models, names, variables, n_cv=1, cohort="merged", HVPG_threshold=16, balance=False,
                          title="", analyze=False, save_folder_AUCs=None, save_folder_plots=None):
    """
    For a given set of independent variables, run CVs and compare prediction performance of the provided models, optionally only for a specific subset (cohort) of the dataset. Returns AUC scores from all CVs and fold for each model per fold, and statistics (mean/median AUC values + confidence intervals). If the balance flag is set, randomly sample the same number of records for both the minority and majority class.

    Parameters
    ----------
    data : DataFrame
        Data table with dependent (HVPG) and independent variables + information about cohorts
    models : list
        List of sklearn models to compare
    names : list
        Names of the models to show in the reported statistics
    variables : list
        Names of the parameters to use to test the performance of models
    n_cv : int
        Number of CVs to perform.
    cohort : string
        Restrict the dataset only to one cohort (subset). If "merged" then use all data
    balance : bool
        Flag to control dataset balancing. If set to true, the same number of patients as is in the minority class is randomly sampled from the majority class
    title : string
        Name of the analysis to display in plots
    analyze : bool
        Flag, if set to true, the AUC performance will be plotted using a swarmplot
    save_folder_AUCs : string
        If set, the AUC scores for each fold will be saved in the specified folder in csv format
    save_folder_plots : string
        If set,  the plots will be saved in the specified folder in pdf format

    """     
    # restrict the dataset to only selected parameters and optionally patients from selected cohort
    if cohort != "merged":
        data_subset = data[data["dataset"] == cohort].copy()
        data_subset = data_subset[["HVPG"] + variables].copy()
    else:
        data_subset = data[["HVPG"] + variables].copy()
    
    # remove patients with incomplete records for the required parameters
    for variable in ["HVPG"] + variables:
        data_subset = data_subset[data_subset[variable].notnull()]
        
    # balance the dataset if desired
    if balance:
        # split the data between low-risk and high-risk
        low_high_risk_separate = (data_subset[data_subset["HVPG"]<HVPG_threshold].copy(), data_subset[data_subset["HVPG"]>=HVPG_threshold].copy())
        # figure out which group is smaller
        larger_grp_idx = np.argmax((low_high_risk_separate[0].shape[0], low_high_risk_separate[1].shape[0]))
        smaller_grp_idx = np.argmin((low_high_risk_separate[0].shape[0], low_high_risk_separate[1].shape[0]))    
        # take all patients from the smaller group and an equal number of random patients from the other group
        n_patients_smaller_grp = low_high_risk_separate[smaller_grp_idx].shape[0]
        data_balanced = low_high_risk_separate[smaller_grp_idx].append(low_high_risk_separate[larger_grp_idx].sample(n=n_patients_smaller_grp))
        data_subset = data_balanced
    # trick to report if there was balancing
    report_dict = {False: "no", True: "after"}

    print(f'HVPG threshold {HVPG_threshold} mmHg, {len(variables)} variables ({", ".join(variables)}), {cohort} cohort(s), {data_subset.shape[0]} patients ({report_dict[balance]} balancing), {n_cv} cross-validation(s)\n')  
            
    X = data_subset[variables]
    y = 1*(data_subset["HVPG"]>=HVPG_threshold)
    
    AUCs, mean_AUCs = repeat_CV_get_AUC(models, names, n_cv, X, y)
    print()
    display(mean_AUCs)
    
    # add info to the title
    if title:
        if HVPG_threshold != 0:
            title = f"HVPG{HVPG_threshold}, " + title
        title += f", {cohort} cohort ("  
        if balance:
            title += "balanced, "
        title += f"n={data_subset.shape[0]})"
        filename = re.sub("[ -]|cohort|n=.*", "", title)
        filename = re.sub("\+|,|\(|\)|_+", "_", filename)
        filename = re.sub("_+", "_", filename)
        filename = re.sub("_$", "", filename)        
    
    if save_folder_AUCs and title:
        AUCs.to_csv(f"{save_folder_AUCs}/{filename}.csv", index=None)
    
    if analyze:
        analyze_mean_AUCs(pd.DataFrame(AUCs), n_cv, title, save_folder_plots)
    
    return AUCs, mean_AUCs


def analyze_mean_AUCs(AUCs, n, title="", save_folder=None):
    """
    For each model, plots mean AUCs from each CV and reports the overall mean AUC.

    Parameters
    ----------
    AUCs : DataFrame
        AUC scores. Rows = individual folds from 5-fold CVs. Columns = models.
    n : int
        Number of CVs performed to show on the plot.
    title : string
        Title to show on the plot. Used also to generate a filename if save_folder is set.
    save_folder : string
        If set, the plots will be saved in the specified folder.

    """      
    AUCs = pd.DataFrame(AUCs)
  
    # prepare labels for the x axis: model names and means of AUCs
    means = AUCs.mean()
    xlabels = [(means.index[i], means[i]) for i in range(len(means))]
    xlabels = [f'{name.replace(" ",chr(10))}\n({val:.3f})' for name,val in xlabels]
    
    ylabel = f"AUCs ({n} CVs)"
    ylim = (.6, 1)
    yticks = np.arange(.65, 1.001, 0.05).round(2)
    
    # plot config (size, style)
    plt.figure(figsize=(6,6))
    #sns.set_theme(context="notebook", style="whitegrid", font="helvetica")
    plt.grid(axis='y', color="lightgray")
    
    # plot dots for means; move this layer to the top
    ax = sns.scatterplot(data=pd.DataFrame({"_":range(len(means)), "__":means}), x="_", y="__", color="black",edgecolor='white', s=50, linewidth=1.1)
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    
    # violin plot - for each CV only the mean
    CV_means = {}
    for variable in AUCs.columns:
        CV_means[variable] = list([np.mean(AUCs[variable][i:i+5]) for i in range(0,len(AUCs), 5)])
    sns.swarmplot(data=pd.DataFrame(CV_means), s=3)
    
    # plot config
    plt.xlabel("")
    plt.ylabel(ylabel, fontweight="bold", labelpad=10, fontsize=16)
    plt.title(title, fontweight="bold", fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks, fontsize=16)
    ax.set_xticklabels(xlabels, fontsize=16)
    if save_folder:
        filename = re.sub("[ -]|cohort|n=.*", "", title)
        filename = re.sub("\+|,|\(|\)|_+", "_", filename)
        filename = re.sub("_+", "_", filename)
        filename = re.sub("_$", "", filename)    
        plt.savefig(f"{save_folder}/{filename}.pdf", bbox_inches="tight")
    plt.show()    
    

def train_one_cohort_validate_rest(data, models, names, variables, train_cohort="VIENNA", plot_cohorts=True, HVPG_thres = 16):
    """
    Train models on one cohort and validate on the rest. Report AUC scores for each model for separate cohorts.

    Parameters
    ----------
    data : DataFrame
        Data table with dependent (HVPG) and independent variables + information about cohorts
    models : list
        List of sklearn models to compare
    names : list
        Names of the models to show in the reported statistics
    variables : list
        Names of the parameters to use to test the performance of models
    train_cohort : string
        Name of the cohort to train on.
    plot_cohorts : string
        Show distribution of labels in the different cohorts for the current parameter setting.

    """   
    plt.rcParams['font.size'] = 14
    data_subset = data[["HVPG", "dataset"] + variables].copy()
    data_subset["HVPG_label"] = f"HVPG < {HVPG_thres}"
    data_subset.loc[data_subset.HVPG >= HVPG_thres, 'HVPG_label'] = f"HVPG ≥ {HVPG_thres}"
    
    
    # remove patients with incomplete records for the required parameters
    for variable in ["HVPG"] + variables:
        data_subset = data_subset[data_subset[variable].notnull()]
    data_subset = data_subset.reset_index(drop=True)
        
    # list cohorts with patients with relevant parameters measured
    cohorts = data_subset["dataset"].drop_duplicates().to_list()
    
    # show some info/stats about the subset of the data
    total_patients = len(data_subset)
    low_risk = 1*(data_subset["HVPG"]>=HVPG_thres).value_counts().iloc[0]
    high_risk = 1*(data_subset["HVPG"]>=HVPG_thres).value_counts().iloc[1]
    
    print(f'{len(variables)} variables ({", ".join(variables)}), trained on: {train_cohort}, {total_patients} patients, {low_risk} HVPG<{HVPG_thres}, {high_risk} HVPG>={HVPG_thres}')  
    train = data_subset[data_subset["dataset"] == train_cohort]
    validate = [data_subset[data_subset["dataset"] == name] for name in cohorts]
    
    if plot_cohorts:
        plt.figure(figsize=(5,5))
        sns.countplot(y = 'dataset', data = data_subset, hue = "HVPG_label").set_title(f'HVPG{HVPG_thres}, patients with {", ".join(variables)} measurements')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.show()
    
    X_train = train[variables]
    y_train = 1*(train["HVPG"]>=HVPG_thres)
    
    X_validate = [cohort_data[variables] for cohort_data in validate]
    y_validate = [1*(cohort_data["HVPG"]>=HVPG_thres) for cohort_data in validate]

    
    result = []

    for j, model in enumerate(models):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # don't show annoying warnings
            model.fit(X_train, y_train)

        scores = []
        for i, cohort in enumerate(cohorts):
            score = roc_auc_score(y_validate[i], model.predict_proba(X_validate[i])[:, 1])
            scores.append(score)
        result.append(scores)

    result = pd.DataFrame(result, columns=cohorts, index = names)
    # reorder
    cohorts.remove(train_cohort)
    cohorts = [train_cohort] + sorted(cohorts)
    result = result[cohorts]
    return result


def plot_ROC_CV(clf, cv, X, y, color='b', model="", lines=['-', '--', '-.', ':']):
    """
    Runs cross-validation on the provided data and plots the ROC curve.

    Parameters
    ----------
    clf : sklearn model
        Classifier
    cv : StratifiedKFold object
        Specifies how to split the dataset into folds
    X : matrix/DataFrame
        Independent variables
    y : vector/DataFramew
        Dependent variable
    color : string
        Color to show on the ROC curve
    model : string
        Name of the model to show on the plot.
    lines : list
        List of strings indicating the line styles for individual folds. Must be of the same length as the number of folds specified in the cv argument.
    """  
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, color=color, linestyle=lines[i],
                 #label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
                 label=f'{model}, fold {i+1} ({roc_auc:.3f})')

        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=color,
             label=f'{model}, mean ({mean_auc:.3f})',
             lw=1.75, alpha=.8)
    