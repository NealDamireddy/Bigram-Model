# Bigram-Learning-Model
This is a learning model using PyTorch that takes the previous letter of a string and uses that to predict the next character.

In this project, our goal was to take a dataset of names, use nn.Embedding to train it, and then build a predictive model that takes in the current character and produces a prediction for the next character. Here is the architecture:

We first have to split our data into testing and training data. What we did was take 70% of the names and convert them into training data, 15% into validation, and 15% into testing.

We then created our vocabulary using strictly our training data to prevent leakage. We pushed it through the build_vocabulary function, where we went through each name, collected all the unique characters, sorted them alphabetically, assigned each character an index, and then returned our vocabulary with each character corresponding to an index in the form of a dictionary.

We then passed each of our training, validation, and testing datasets separately into the build_dataset function. This took the names and split them into 2 tensors: one tensor containing the current character's index and the other containing the next character's index. For example, if we take ".emma.", our X tensor would represent [".", "e", "m", "m", "a"] and our Y tensor would represent ["e", "m", "m", "a", "."], with each character converted into its corresponding integer ID.

After this, we then built our model architecture. We created a class inheriting from torch.nn.Module and initialized an embedding matrix using the length of our vocabulary dictionary (stoi). In this case, both dimensions are determined by the vocabulary size. Each row represents a character, assigned by its index, and contains 27 raw logits (assuming our vocabulary has 27 characters), where each logit corresponds to a possible next character.

For example, if we passed through [0] into this model, it would correspond to itos[0], which is ".". Our nn.Embedding would retrieve a 27-dimensional vector that includes the raw logits for the next-letter prediction. We then compare these logits against our Y tensor using cross-entropy loss during training.

We then iterated this model using our training dataset with a basic optimizer and backpropagation, while cross-referencing the validation dataset to check for overfitting. These were our preliminary results:

Epoch 100 | Train loss: 2.9187 | Validation loss: 2.9085
Epoch 200 | Train loss: 2.6140 | Validation loss: 2.6086
Epoch 300 | Train loss: 2.5145 | Validation loss: 2.5107
Epoch 400 | Train loss: 2.4820 | Validation loss: 2.4789
Epoch 500 | Train loss: 2.4700 | Validation loss: 2.4672
Epoch 600 | Train loss: 2.4645 | Validation loss: 2.4617
Epoch 700 | Train loss: 2.4615 | Validation loss: 2.4587
Epoch 800 | Train loss: 2.4595 | Validation loss: 2.4568
Epoch 900 | Train loss: 2.4583 | Validation loss: 2.4555
Epoch 1000 | Train loss: 2.4574 | Validation loss: 2.4546
Epoch 1100 | Train loss: 2.4567 | Validation loss: 2.4539
Epoch 1200 | Train loss: 2.4562 | Validation loss: 2.4535
Epoch 1300 | Train loss: 2.4558 | Validation loss: 2.4531
Epoch 1400 | Train loss: 2.4555 | Validation loss: 2.4528

Lastly, the most important part was the sequence generation. I used an autoregressive function denoted by generate_sequence(model, stoi, itos, max_length=20). This function was the hardest part for me.

First, I switched the model to eval mode while also creating a list that contained our predicted characters.

I then created a tensor that takes in the current character's index and converts it into a tensor of dtype torch.long.

I then moved to a for loop, where we run this tensor X through our model (remember, X is just a tensor that tells the embedding matrix which index we want to look at).

This then returns a 27-dimensional vector where each logit represents a score for a possible next character, and we use the softmax() function to convert these into probabilities.

After we convert it to probabilities, we use torch.multinomial() to select an index proportionally to each probability assigned by the model. This then returns us an index of our prediction, and we just use itos[next_index] to find our next character prediction.

As long as this prediction doesn't give us index 0 (which corresponds to ".", indicating the end), then we add this predicted character to the list of characters we initialized earlier. We then update our current character's index and recreate the input tensor using this new index, iterating over that until we reach max_length or ".". We join the list of characters and return the string.

Running this:

nni
mizhi
kamickasilanyaun
a
kh
anahamoryrial
ja
jarelin
lya
ryrvaylyuri

One limitation of this model is that it only takes into account the current character when predicting the next one. This means it can learn common character combinations, but it cannot account for the earlier characters in a name, which limits how realistic the generated names can be.
