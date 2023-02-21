import pickle
from keras.preprocessing.sequence import pad_sequences

def extract_single_question_feature(question, tokenizer_file, padded_sequences_file, maxlen=100):
    # Load the tokenizer
    with open(tokenizer_file, "rb") as file:
        tokenizer = pickle.load(file)

    # Load the padded sequences
    with open(padded_sequences_file, "rb") as file:
        padded_sequences = pickle.load(file)

    # Encode the question
    sequence = tokenizer.texts_to_sequences([question])

    # Pad the encoded question
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)

    # Extract the feature for the single question
    single_question_feature = padded_sequence[0]
    
    return single_question_feature

print(len(extract_single_question_feature('Is this lung?','tokenizer.pkl','padded_sequences.pkl')))