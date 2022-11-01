# Medical  visual  question  answering  via  corresponding  feature  fusion  combined with semantic attention

    CFF method to classify input images and input questions into specific categories. 
    A light weigt CNN module to extract image features
    Word Embedding followed by LSTM to extract question features , pass through SA module to obtain attention weight
    Both Features fused , passed through vqa classifier

    Slake obatined 82.4 accuracy in CFF+SA+CMSA