AlkaneBoilingPoints This simple code plots the number of carbon atoms in the alkane on the x-axis and the boiling points of
that alkane on the y-axis. It generates a properly labeled graph
using Matplotlib.
This is part of the CobberLearnChem Machine Learning course
through Concordia College but taken at Roosevelt Univeristy.

PubChemFetcher is a code designed to pull particular chemical information from PUBCHEM without needing to download files. 

MoleculeExplorer is a code designed to pull the SMILES code string out and place it into RDKit for analysis. It should allow the user to look at multiple molecular discriptors without restarting the program.

The code developed in Ch.6 Analysis Questions Part 1 - 5 was used to Answer the analysis questions with the textbook peppered throughout the Sixth chapter. 

The code developed in Ch. 5 Analysis Questions was use to answer the analysis questions within the textbook.

The code in CobblerImpute was developed to perfrom a listwise deletion and see how it impacts the bias in the data. The file labeled listwise_deletion_bias.png is the plot generated from this code. The following CobberImpute 2-5 run different imputations (KNN, RF, Log-linear regression, and Ensemble Model) to compare them to the listwise deletions.

The MakingDataWhole code takes the data from the Titanic dataset to run similar code to cobber Impute, but specifically for KNN and RF imputations. 

The CobberResidue codes walk through the importation of multiple datasets to compare MAE, MSE, and R squared of multiple types of datasets and how this values are impacted.

The ErrorMetrics code performs a similar task to CobberResidue, but uses NumPy to create arrays from which plots and the MAE, MSE, and R squared values are calculated from. 

The code written in ElementsClustering and ElementClustering 2 provides the ability to import a csv file of group 1 and group 2 elements of the periodic table - inlcuding name, symbol, atomic number, atomic radius, and first ionization energy - and uses that data to create a scatterplot. There is a loop that allows the user to input a value for K and cluster the data.

The code written in Gradient Descent is creating a noisy dataset using numpy, then generates a scatter plot and a fitted line and then generates a loss landscape visualization. 

The METHINKS IT IS LIKE A WEASEL code directory creates code that is similar to the famed "The Weasel Program" and creates a string that can be modified to match the target phrase. It uses an evolution loop to develop the string and slightly modify it over time, scoring it and replacing the parent string if the string is better. In addition, there is a text box added to change the target phrase if desired. 

The code written in Androgen Angonist Pipowski works to import 4 datasets into python and combine them into a master dataset which can exported as a .csv file for use. The purpose of the code is to work on our class's final project. 

The code written in AR Agonist Pipowski_2 takes a LUC Assay and LUC Viability Assay and combines them into a master dataset named "LUC_Master_Dataset." The LUC Assay tests for luciferase report gene activity via bioluminescence signals. The Viability Assay uses the LUC Assay and uses the Nilutamide viability to look for loss-of-function to understand changes in cell viability. For each column in the dataset, I will explain what they represent below. 
  - DTXSID: The unique id code given by CompTox to each chemical, used to organize each assay by chemical.
  - Preferred Name: This column is simply the name of the chemical.
  - CASRN: This column is the Chemical Abstracts Service Registry Number, which is unique identifier that links the chemical to its chemical structure data, its bioactivity, and regulatory information.
  - Molecular Formula: This is the molecular formula of the molecule, but NOT a smiles string.
  - Monoisotopic Mass: This column showcases the average mass of the chemical by averaging all known isotopes of the constitute elements.
  - TOXCAST ACTIVE_Agonist: This column showcases the total number of ToxCast Endpoints where the sample is considered active in the LUC_Agonist assay.
  - TOXCAST_TOTAL_Agonist: This column represents the total number of Toxcast endpoints where the chemical was screened.
  - % TOXCAST ACTIVE_Agonist: This column is the percentage of active samples across all ToxCast endpoints where it was screened.
  - HIT CALL_Agonist: This column tells us if the LUC Assay determined the chemical in question to be active or inactive. If active, the 
  - CONTINUOUS HIT CALL_Agonist: This column showcases if the chemical, if active, reacted in the intended response. A negative value means it reacted in an unexpected way.
  - TOP_Agonist: This column is a value that represents the best model where maximum response was observed relative to the control.
  - SCALED TOP_Agonist: This column is a value that is calcualted by dividing response values by the activity cutoff, which allows for response comparisons across different assay endpoints.
  - AC50_Agonist: This column represents the activity concentration at 50% of maximal activity. A lower value indiciates that a chemcial is more potent and a lower concentration is needed to achieve half of the maximum observed response in the Assay.
  - LOGAC50_Agonist: This column is the linearization of the AC50 for the purpose of regression calculations and comparisons.
  - TOXCAST ACTIVE_Viability: This column represents the number of ToxCast Endpoints where the sample is considered Active in the Viability assay.
  - TOXCAST TOTAL_Viabilty: This comun represents the total number of ToxCast endpoints where the chemical was screened.
  - % TOXCAST ACTIVE_Viability: This column is the percentage of active samples across all ToxCast Endpoints where the chemical was screened.
  - HIT CALL_Viability: This is the column that tells us if the chemical was active or inactive for the Viability Assay. If active, the cell is alive and metabolically active. If inactive, the cell is dead. 
  - CONTINUOUS HIT CALL_Viabilty: This column represents if an actives chemcial reacted in the expected response. A negative value means the chemical reacted in an unexpected way.
  - TOP_Viability: This column is a value that represents the best model where maximum response was observed relative to the control.
  - SCALED TOP_Viability: This column is a value that is calculated by dividing response values by the activity cutoff, which allows for response comparisons across different Assay Endpoints.
  - AC50_Viability: This column represents the activity concentration at 50% of the maximal activity. A lower value indicates that a chemical is more potent and a lower concentration is needed to achieve hald of the maximum observed response in the Assay.
  - LOGAC50_Viabilty: This column is the linearization of the AC50 for the purpose of regression calculations and comparisons. 

Alongside the assigned coding projects, I will be keeping a private ethics portfolio where I reflect on my learning and how it impacts the kind of scientists I am developing into. 
