#!/usr/bin/env python
# coding: utf-8
import torch
import random
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import time
import datetime
from transformers import get_linear_schedule_with_warmup

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# cd drive/My Drive/CSCI566_project/memes900k

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


selected = set(['african children dancing', 'Angry Cat Meme', "Anchorman Birthday", "Annoying Gamer Kid", "Awkward Seal",
            "Black Kid", "burning house girl", "Desk Flip Rage Guy", "evil plan kid", "FU*CK THAT GUY", "Grumpy Cat 2",
            "Grumpy Cat Santa Hat", "high/drunk guy", "Joseph Ducreux", "Skeptical african kid", "so doge",
            "Success Kid", "Willy Wonka", "You shall not pass", "Y U No"])

x_train = []
y_train = []
# select from github dataset
with open("../memes_data/memes900k/"+"captions_train.txt") as file:
    for line in file.readlines():
        l, _, text = line.split('\t')
        if l.strip() in selected:
            y_train.append(l.strip())
            text = text.strip().replace('<sep>', ", ")
            x_train.append(text)
# x_train = x_train[:5000]
# y_train = y_train[:5000]

# select from https://imgflip.com/memetemplates
# with open("memes_output.txt") as file:
#     for line in file.readlines():
#         l, _, text = line.split('\t')
#         y_train.append(l.strip())
#         text = text.strip().replace(';', "[SEP]")
#         x_train.append(text)

templates_map = dict(zip(set(y_train), range(300)))
templates_map_reverse = dict(zip(range(300), set(y_train)))
len(templates_map)

y_train = [templates_map[i] for i in y_train if i]
# # Build Model

class BertClassification(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self, train_dataloader, validation_dataloader, epochs=4):
        self.model.cuda()
        optimizer = AdamW(self.model.parameters(),
                          lr=7e-5,  # args.learning_rate - default is 5e-5
                          eps=2e-8  # args.adam_epsilon  - default is 1e-8.
                          )
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        seed_val = 66
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        self.training_stats = []

        total_t0 = time.time()

        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                self.model.zero_grad()

                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # Validation

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    (loss, logits) = self.model(b_input_ids,
                                                token_type_ids=None,
                                                attention_mask=b_input_mask,
                                                labels=b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            self.training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))

    def predict(self, validation_dataloader):
        # print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        self.predictions = []

        # Predict
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, _ = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            # label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            self.predictions.append(logits)

        print('    DONE.')
    def single_predict(self, input_ids, input_mask):
        self.model.eval()
        outputs = self.model(input_ids, token_type_ids=None,
                                     attention_mask=input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        return logits

# test_sent = "eating dinner with a friend's family, family leaves for a few minutes"
test_sent = "at a sleepover in a friend's house first one to wake up"
# test_sent = "i am not angry i'm happiness challenged"
encoded_dict = tokenizer.encode_plus(
        test_sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=50,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
bc = BertClassification(25)
bc.eval(bc.load_state_dict(torch.load("pretrained/meme_template_selector.pt", map_location="cpu")))
out = bc.single_predict(torch.cat([encoded_dict['input_ids']], dim=0).cuda(), torch.cat([encoded_dict['attention_mask']], dim=0).cuda())
print(templates_map_reverse[out.argmax(axis=1)[0]])
