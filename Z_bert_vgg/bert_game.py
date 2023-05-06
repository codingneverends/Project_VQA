from transformers import pipeline

# Load the pre-trained BERT model for question answering
qa_model = pipeline("question-answering", model="bert-base-uncased", tokenizer="bert-base-uncased")

# Define the context and question
context = "Rama Killed a Snake"
question = "Who killed Snake?"

# Use the BERT model to answer the question based on the context
result = qa_model(question=question, context=context)

# Print the answer
print("Context : ",context)
print("Question : ",question)
print("Predicted Output : ",result["answer"])
