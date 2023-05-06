# Import the necessary modules
import torch
from transformers import BertTokenizer, VisualBertConfig, VisualBertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
# Define a custom model class that inherits from VisualBertModel
class CustomVisualBertModel(VisualBertModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # You can add any custom layers or attributes here
        # For example, you can add a dropout layer and a classifier layer
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, inputs):
        # You can override the forward method of the parent class and add any custom logic here
        # For example, you can apply the dropout and classifier layers to the pooled output
        outputs = super().forward(inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Define a custom dataset class that inherits from Dataset
class CustomDataset(Dataset):
    def __init__(self, questions, images, labels, tokenizer):
        # Store the questions, images, and labels as tensors
        self.questions = questions
        self.images = torch.tensor(images)
        # Convert the labels to one-hot vectors using torch.nn.functional.one_hot
        self.labels = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64), num_classes=2)
        # Store the tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the length of the dataset
        return len(self.questions)

    def __getitem__(self, idx):
        # Get the question, image, and label at the given index
        question = self.questions[idx]
        image = self.images[idx]
        label = self.labels[idx]
        # Encode the question and image using the tokenizer
        encoding = self.tokenizer(text=question, images=image, return_tensors="pt", padding=True)
        # Return the encoding and label as a tuple
        return encoding, label

# Instantiate the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa", num_labels=2)
model = CustomVisualBertModel.from_pretrained("uclanlp/visualbert-vqa", config=config)

# Prepare some sample data (you can replace this with your own data)
# Getting Data to Train
import json
# Get data from existing dataset
train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
train_qns = json.load(train_qns)
train_qns = [x for x in train_qns if x['content_type']=='Modality' and (x['answer']=='Yes' or x['answer']=="No")]
questions = [x['question'] for x in train_qns]
image_names=["../Dataset/Slake/imgs/"+x['img_name'] for x in train_qns]
img_width,img_height=224,224
images = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in image_names])
answers = np.array([1 if x['answer'] == 'Yes' else 0 for x in train_qns])

# Create a dataset object from the questions, images, and labels
dataset = CustomDataset(questions, images, answers, tokenizer)

# Create a data loader object from the dataset object
loader = DataLoader(dataset, batch_size=16)

# Define an optimizer and a metric function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
metric_fn = accuracy_score

# Define the number of epochs
epochs = 3

# Train the model on the data
for epoch in range(epochs):
    # Set the model to training mode
    model.train()
    # Initialize the loss and metric values
    train_loss = 0.0
    train_metric = 0.0
    # Loop over the batches of data
    for batch in loader:
        # Get the inputs and labels from the batch
        inputs, labels = batch
        print(inputs[0],labels[0])
        # Forward pass the inputs to the model and get the outputs
        outputs = model(inputs)
        # Compute the loss using torch.nn.functional.cross_entropy with one-hot labels
        loss = torch.nn.functional.cross_entropy(outputs.view(-1).float(), labels.float())
        # Backward pass the loss and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate the loss and metric values
        train_loss += loss.item() / len(loader)
        # Use torch.argmax to get the class index from one-hot labels and outputs before passing them to metric function
        train_metric += metric_fn(torch.argmax(labels.numpy(), axis=1), torch.argmax(outputs.detach().numpy(), axis=1)) / len(loader)
    # Print the epoch summary
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Metric: {train_metric:.4f}")
