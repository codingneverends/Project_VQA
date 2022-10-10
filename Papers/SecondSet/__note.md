# BPI‑MVQA: a bi‑branch model for medical visual question answering

Bi-branched model -- Parallel networks and Image retrieval -- BPI-MVQA

First Branch 
transformer structure as the main framework of parallel structure model.
we adopt a parallel network to extract the image features. Firstly, we use an
improved CNN model to extract the spatial features of
the medical images. Secondly, we use an RNN model to
extract the sequence features of the medical images.
Text feature also extracted.
We embed the above-mentioned image features into the front part of the question(text) features,
integrate the two parts of features into a feature matrix, and then input it into the stacked four-layer transformer structure. As a result, the model can learn
the dependency between image features and question
features, and capture the internal structure of
the input feature vector.

Second branch
we use the answers of the training set as
the labels of the corresponding images, ignoring the influence
of question features on the classification results.

Achives state-of-art result of 3 VQA-Med(ImageCLEF2018,ImageCLEF2019,VQA-RAD) datasets..main metric score excees by 0.2%,1.4%,1.1%

# Hybrid deep learning model for answering visual medical questions
