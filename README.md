# TARnSIHED-WW

## Project Organization

```         
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data               <- The original, raw data and processed data
│
├── models             <- model results, model checkpoints, or model summaries
│
├── notebooks          <- Main pipeline of models.
│
├── pyproject.toml     <- Project configuration file with package metadata for TARnSIHED_WW and configuration for tools like black
|
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── paper            <- Generated manuscript as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── TARnISHED_WW   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes TARnISHED_WW a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── build_functions.py              <- Code to create bayesian modules for TARnISHED_WW
    │
    ├── helper_functions.py             <- Code to create features, metrics and others for models
    │
    ├── forecast_functions.py        <- Code to create TARnISHED forecasts
    │      
    ├── plots_functions.py                <- Code to create visualizations
    └── training.py                <- Code to train XGBoost or other ML models
```

------------------------------------------------------------------------

# Wastewater Predictive Modelling of Disease Outcomes

## Description

This project main goal is to develop a predictive model of emergency department visits due to respiratory illness using wastewater data as an extra information in the time series. This project has many subprojects:

1. Predict ED visits for each WWTP using the viraload of multiple respiratory virus (RSV, covid-19, and Influenza A).
2. Predict Reported Cases for each WWTP and each virus.
3. Predict hospitalizations for each WWTP and each virus.
4. Design new features based on genomic surveillance on wastewater (Covid-19 only).
   
Models that were implemented: Poisson Regression, Lienar Regression.
Models currently being implemented/fine-tuned: XGBoost, GRUs, NeuralNets, ARIMAX
In the notebooks folder, we can find some of these implementations.

Next steps, rewrite the notebook functions into .py scripts to automate the training/validations and prediction.

## Notebooks

**Current working models**

- *01data-processing.ipynb* cleans and process data on the raw and raw_query folder to the final input data in the processed folder. Also fetchs weather data.

- *XGBoostRegressor.ipynb* trains a XGBoost on dataframe (regions, target, feature 1, feature2, feture2, ...)

- *GRU_pytorch.ipynb* trains a Gated Recurrent Unit on dataframe (regions, target, feature 1, feature2, feture2, ...).

## Project status

In progress:

**Project Progress: 40%**

🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜  40%


## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

-   [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
-   [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```         
cd existing_repo
git remote add origin http://lvmgenodh01.phsabc.ehcnet.ca/bccdc/math_modeling/wastewaterpred.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

-   [ ] [Set up project integrations](http://lvmgenodh01.phsabc.ehcnet.ca/bccdc/math_modeling/wastewaterpred/-/settings/integrations)

## Collaborate with your team

-   [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
-   [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
-   [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
-   [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
-   [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

-   [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
-   [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
-   [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
-   [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
-   [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

------------------------------------------------------------------------

## Contributing

State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.

## License

For open source projects, say how it is licensed.
