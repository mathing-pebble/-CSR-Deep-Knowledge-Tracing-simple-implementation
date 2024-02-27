The following is an implementation of the RNN-based DKT model introduced in Deep Knowledge Tracing by Chris Piech et al. More information about the model can be found in the paper below.
(https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
The datasets used for the implementation are as follows:

builder_train.csv / builder_test.csv: The data must be interpreted in sets of three rows.
The first row is the number of answers (n_answer), the second row is the question_id, and the third row is whether the trial corresponding to the question_id is correct.

More detailed information about the data can be found at the link below.
https://sites.google.com/site/assistmentsdata/datasets/2011-goldstein-baker-heffernan?authuser=0

The files written for model implementation are as follows:
dataAssist.py: This is the process of preprocessing the datasets used for training and evaluation through the dataMatrix class.
rnn.py, LSTM.py : RNN/LSTM layer used for training the model is implemented in the files 'rnn.py', 'lstm.py'
trainAssist.py : Using 'dataAssist.py' and 'LSTM.py', we implement the training process of the DKT model introduced in the paper mentioned at first. 

To run the model, it is required to adjsut the 'root' variable in 'dataAssist.py', defined at line 8 with the address of csv.file.
