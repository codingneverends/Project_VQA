### Abbr
    BLEU (BiLingual Evaluation Understudy) is a metric for automatically evaluating machine-translated text.

    WBSS computes a similarity score between a system- generated answer and the ground truth answer based on word-level similarity.

BERT -  Bidirectional Encoder Representations from Transformers

CNN - Convlutional NN

EfficientNET - convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient.

RNN - Recurrent NN -- low memmory -- can create loops

LSTM - Long Short Term Memmory -- Contains memmory cell -- better than RNN

Bi-LSTM - Bidirectional LSTM -- more better

VGG - Visual Geometry Group - standard deep CNN


### How VQA Done!

    1.BERT Model on question.
        4 categories of questions plane,modality,organ,abnormality
    2.Image Feature extraction using CNN,EfficientNet.
    3.Question Feature extraction using RNN/LSTM/Bi-LSTM.
    4.Fuse Image and Question Features to obtain answer using ML algos(Genetic Algo,DNN,...).
