# Programming for Data Analysis Project 2022

**Version 1.0.0**

## Description
This is the analysis of Breast Cancer Wisconsin Data Set using different statistics, visulalisation methods and machine learning models in Python.

## A list of items in the Programming for Data Analysis Project 2022:
- DataFrame background and overview
- Checks for statistical analysis
- Summary and basic statistics
- DataFrame visualisation
- Testing DataFrame by creating 5 different machine learning models:
    - LogisticRegression
    - LinearRegression
    - RandomForestClassifier
    - GaussianNB
    - KNeighborsClassifier
- Results Comparision

## How to use these
1. Make sure that Python including Jupyter is installed on your machine (you can easily download Python via Anaconda from the internet).
2. Go to my Github repository: https://github.com/Anna20041983/PfDA-Project
3. Click the download button to save a copy of the repository on your machine.
4. In CMDER or any other command line type 'Jupyter Notebook' and click enter.
5. Once the Jupyter Notebook is opened in your default browser, go to each individual file (from Week 1 to Week 5 in Practicals Folder and Normal Distribution Notebook)
6. In each individual Jupyter Notebook click on 'Kernel', then 'Restart & Run All' to run codes and generate graphs

## Inspiration
- DataSet download: https://data.world/health/breast-cancer-wisconsin
- DataSet overview: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
- Preparation for analysis and visualisation:
    - https://sparkbyexamples.com/pandas/pandas-replace-substring-in-dataframe/
    - https://www.tutorialspoint.com/matplotlib/matplotlib_pie_chart.htm
    - https://medium.com/geekculture/create-a-pie-chart-in-python-using-only-one-line-of-code-57bd974d8432
    - https://stackoverflow.com/questions/7082345/how-to-set-the-labels-size-on-a-pie-chart-in-python
    - https://www.python-graph-gallery.com/92-control-color-in-seaborn-heatmaps
    - https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
    - https://discuss.streamlit.io/t/change-the-font-size-of-labels-in-sns-heatmap/35454/2
    - https://stackoverflow.com/questions/35420642/how-to-plot-a-graph-for-correlation-co-efficient-between-each-attributes-of-a-da
    - https://www.kaggle.com/code/swagata14das/breast-cancer-prediction
- Machine learning:
    - https://stackoverflow.com/questions/40353079/pandas-how-to-check-dtype-for-all-columns-in-a-dataframe
    - https://www.ritchieng.com/pandas-changing-datatype/
    - https://stackoverflow.com/questions/8924173/how-can-i-print-bold-text-in-python
    - https://www.simplilearn.com/tutorials/scikit-learn-tutorial/sklearn-linear-regression-with-examples
    - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
    - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    - https://stackoverflow.com/questions/68799909/classification-accuracy-with-sklearn-in-percentage
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    - https://careerkarma.com/blog/python-round/
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    - https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    - https://matplotlib.org/2.0.2/users/gridspec.html
    - https://lifewithdata.com/2022/02/06/confusion-matrix-how-to-plot-and-interpret-confusion-matrix/
    - https://www.jcchouinard.com/classification-report-in-scikit-learn/
    - https://www.earthdatascience.org/courses/intro-to-earth-data-science/file-formats/use-text-files/format-text-with-markdown-jupyter-notebook/
- Comparision with external analysis:
    - https://www.hindawi.com/journals/abb/2022/6187275/
    - https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050918X00088/1-s2.0-S1877050918309323/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEQaCXVzLWVhc3QtMSJHMEUCIDRiAUNFMkIvTNlINHtwZSk1WcJbwfrCxRUDwvGHRvtlAiEAs6QMYuiCCgXMcEyyaaJtJWrxTJlvUVvzlFezIskLRkIq1QQInf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDDAd4w0%2F59FcnfC8%2FSqpBFucVTo%2FpYa8lvmUKIr%2BsDlKOL1biR0PbYaH%2BswWFF669Z8QpN7Nn3C5U1f9eAsL0iahmy90n3n2dYfLTs0vI%2BZGicXlTVfuZmmJIQcRwEN3u8bclGqsiLxO1l6CWr603n1I9DgOk936ZbpaboQfM6Dkjk9QYe8SOkhbdWD4%2BNl2e24dR1liWY3cOWgg9nJWEYmYFlXXGEG9142%2FFMYctRyG%2F%2Fukl4t34pX2FJHuE77BQt8xNih9Z9Mdjv1iaRFwJQE5jbJ5fNkWPRBA0%2FNsPysocX80kXH6UQtgbqjbepQOcmIMPK2f87%2BoPWntmbXNY%2Fe0SPCJCoUBkTaXcp4crrNppzSSLQStcQ1kWuImGHbZbxW2BIyKXXhmr14PR3LBw86Hsx8ZswVF1QaN12C%2B1C2StXkFuWQmfILW14m5Rv%2F9l7cRoIVoOTCQ6tPEYMV3C6ZKFjXgPCuj9APCgaX8YV7Rv%2FBQpMhEig00oqXTlIpirOSEw7DR%2Btu4D0WWhgBCNHRod%2FwvD3j8Sbf8Mw4bc4WgAIkHKZ%2FwUh3x8H86HU62yBgDsM13E69drOlOT%2BQlmVUFPpnCJu9dpYy1JvZp0TrYWBXgKzLt5j5phO5%2B1fcRNo48t2qWXNJTHF7B81ZQvHvFj0p4p0nUs20R1s%2FJneeFFWTeMtI0CAOqN6X3NBPEp6KiUdYNRtHGsx%2BVgOo0D5KgADmTjoO4OIBGi4G%2BOiIZNZslCyrW740whuf4nQY6qQHQM45oVb4aAxtB97xM66Jf6CXOi%2BemoH0kDPLBGkSmTjgtoUMg%2Fn38Azj7bksL3tdj9z%2B40yDEZFagMPRxqpLIQ%2FZ2MNUR0DTlOKtRrocY%2F2hyaCvIgrwFZ1mKIAFBxggXyfwoM9FGyy4eSGJ2xqwexiNKipfK0GMomVckMnXeVfPBjQV2qd%2Fv%2FYozmaotLDDMq9bJUc1zE4W9DMl8PW1UP3LTjsMCNdng&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230111T044851Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6NYFYUHK%2F20230111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c99a847da966372a7c3772f0637a2405f4fa51ee1d3096a7beaea17c5950797f&hash=a2802d4ca0bd3d2be8ac56ddf9cd2150a36626edaa2718afd994285619a77973&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050918309323&tid=spdf-cd1d5b3c-c90d-4144-ab9b-c42994eb7d31&sid=c2775df449c4854edc6a90a202081fbc4b53gxrqb&type=client&ua=5157510d565606000006&rr=787b05bdac8f7743
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7330506/
- How the Data could be extended: https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html

## Author

Anna Kozakiewicz
