# Rising flanks of calcium transcient

### This work is solely for calcium imaging data.

Calcium flows into the soma of neurons thus activating the green fluorescence protein (GFP), which then fluoresce. Its known from the characteristics of the GFP, it has a very fast rise but slow fall. It is essential to understand that the pumping out of calcium from the soma does not have any contribution to the causal relation between any two neurons.

![image](https://user-images.githubusercontent.com/47278559/209997772-998b1c87-c6ff-463b-9162-d960484d3e82.png)

Here in this project, I will be exploring only the activations of the neurons and find meaningful structures in the data using an algorithm I have developed.


This algorithm has been detailed in the paper ...


In other to be able to select appropriate association for any pair of neuron
- We can start with outliers detection algorithm which performs best on the data as shown in [here](https://github.com/adedayoas91/risingFlanks/blob/main/plot_anomaly_comparison.ipynb) to remove the outliers which might be as a result of artefact in data.
- We can then use the procedures in [Classification of Scatter Plot Images Using Deep Learning](https://dergipark.org.tr/tr/download/article-file/1910064) and [ScagCNN](https://phamvanvung.github.io/publications/ScagCNN.pdf) to identify the shape of data points in scatterplots that actually coordinate to a proper correlation.  We can then run the CNN framework to detect the shape of the data.
