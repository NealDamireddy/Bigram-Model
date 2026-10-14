import torch
import numpy as np
from sklearn.model_selection import train_test_split
with open("/Users/nealdamireddy/tiny_language_model/names.txt", "r") as f:
    names = f.read().splitlines()

class BigramLearningModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, vocab_size) #our output is one of the 27 characters we initalized so both parameters are 27.
        #each row is a vector representation of an id, - each feature of the row represents the probability of the next character    
    def forward(self, X):
        logits = self.embed(X) #raw prediciton scores 
        return logits
    
    

def build_vocabulary(names):
    unique_chars = set() #this creates a set, which automatically removes duplicates
    stoi = {"." : 0} #the first value in stoi should be the .
    for name in names:
        unique_chars.update(name) #takes the string, turns it intosize  characters, and updates on the non-duplicate characters
    unique_chars = sorted(unique_chars) #sort them by index in alphabet, outside of loop because we only do this after we have all of the characters
    for i, chars in enumerate(unique_chars, start = 1): #use enumerate function to assign a value to every character in the list
        stoi[chars] = i #append any preceding values
    itos = {value: key for key, value in stoi.items()}
    return stoi, itos
def build_dataset(names, stoi):
    xs = [] #input list - should be in numbers relating to stoi
    ys = []
    for name in names: #iterate through every name in the dataset
        name = "." + name + "." #add the string
        for i in range(len(name) - 1): 
            xs.append(stoi[name[i]]) #we want to get the index of the name which returns a character and then match that with our integer we assigned in stoi
            ys.append(stoi[name[i +1]])
    X = torch.tensor(xs).long()
    Y = torch.tensor(ys).long()
    return X, Y
names_train, names_temp = train_test_split(
    names,
    test_size = 0.3,
    random_state = 42
    )
stoi, itos = build_vocabulary(names_train)
X_train, y_train = build_dataset(names_train, stoi)
names_validation, names_test = train_test_split(
    names_temp,
    test_size = 0.5,
    random_state = 42
    )
X_validation, y_validation = build_dataset(names_validation, stoi)
X_test, y_test = build_dataset(names_test, stoi)
model = BigramLearningModel(len(stoi))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1500):
    model.train()
    logits = model(X_train)
    train_loss = torch.nn.functional.cross_entropy(logits, y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(X_validation)
            validation_loss = torch.nn.functional.cross_entropy(
                logits, y_validation
            )
        print(
            f"Epoch {epoch} | "
            f"Train loss: {train_loss.item():.4f} | "
            f"Validation loss: {validation_loss.item():.4f}"
        )
#the function of the generate sequence funciton is to take in a dictionary of characters. The important feature of this dictionary is the keys represent the next character. so we want to know the first and next character.
#after we get the current and next character, we want to make sure that the next character is not "." because that would signal the end
#if it isn't then we can append the next character to the list of characters that we initialized before the loop. after we add this, we want to update our current character to the next character


def generate_sequence(model, stoi, itos, max_length = 20):
    generated_list = []
    model.eval()
    current_id = stoi["."] #this will give you the index of the current id
    X = torch.tensor([current_id]).long() 
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(X) #this will create a vector of values, where each feature of this vector is a character.
            #convert tensor to probabilities
            probabilities = torch.softmax(logits, dim = -1) #we have a 27 dimensional vector that takes in our current ID and has a certain logit for each character. we then want to apply softmax to get a probability for each character
            next_index = (torch.multinomial(probabilities, num_samples = 1)).item() #takes the vector of probabilities and selects a random one using the probabilities - this returns us the index of a random next character
            next_char = itos[next_index] #we have the index of the sample that we want. we then take the itos which is the reversed of stoi and find the character corresponding to this index
            if next_char != ".":
                generated_list.append(next_char)
                current_id = next_index
                X = torch.tensor([current_id]).long() #make sure to update the tensor for future iterations
            else:
                break
    return "".join(generated_list)
for _ in range(10):
    print(generate_sequence(model, stoi, itos))      
        
    
            
                    
        
         
    
