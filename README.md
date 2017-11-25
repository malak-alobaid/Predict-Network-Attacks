# Predict-Network-Attacks
Model to Successfully predict Attack type
Lately, an internet network company in Japan has been facing huge losses due to malicious server attacks. They've encountered breach in data security, reduced data transfer speed and intermittent breakdowns in user-user & user-network connections.

When asked, a company official said, " there's a significant dip in the number of active users on our network ".

The company is looking are some predictive analytics solution to help them understand, detect and counter the attacks and make their network connection secure.

Think of a connection as a sequence of TCP packets starting and ending at some well defined times, between which data flows to and from a source IP address to a target IP address under some well defined protocol.

In total, there are 40 type of attacks to which their network is vulnerable to. But, 3 of them cause the maximum damage. In this challenge, you are given an anonymised sample dataset of server connections. You have to predict the type of attack(s).

Download Dataset

Data Description

You are given three files to download: train, test and samplesubmission. The data contains anonymised set of features. Train data has 169307 rows and test data has 91166.

Variable	Description
connection_id	unique id
cont_x	numeric features
cat_x	categorical features
target	target variable (0, 1, 2 are attack types)
