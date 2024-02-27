from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from dataAssist import DataAssistMatrix
from lstm import LSTMModel

START_EPOCH = 1
OUTPUT_DIR = '../output/trainRNNAssist/'
LEARNING_RATES = [30, 30, 30, 10, 10, 10, 5, 5, 5]
LEARNING_RATE_REPEATS = 4
MIN_LEARNING_RATE = 1
NUM_EPOCHS = 10

def run(file_id):
    assert file_id is not None

    np.random.seed(int(time.time()))

    data = DataAssistMatrix(params={})  
    n_hidden = 200
    init_rate = 30
    mini_batch_size = 100
    dropout_pred = True
    max_grad = 5e-5

    print('n_hidden', n_hidden)
    print('init_rate', init_rate)
    print('mini_batch_size', mini_batch_size)
    print('dropoutPred', dropout_pred)
    print('maxGrad', max_grad)

    rnn = LSTMModel({
        'dropoutPred': dropout_pred,
        'n_hidden': n_hidden,
        'n_questions': data.n_questions,
        'maxGrad': max_grad,
        'maxSteps': 4290,
        'compressedSensing': True,
        'compressedDim': 100
    })
    print('rnn made!')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'models/')):
        os.makedirs(os.path.join(OUTPUT_DIR, 'models/'))
    file_path = os.path.join(OUTPUT_DIR, f'{file_id}.txt')

    with open(file_path, "w") as file:
        file.write(f'n_hidden,{n_hidden}\n')
        file.write(f'init_rate,{init_rate}\n')
        file.write('-----\n')
        file.write('i\taverageErr\trate\tclock\n')

        train_mini_batch(rnn, data, mini_batch_size, file, file_id, init_rate)

def create_mini_batches(data, batch_size, shuffle=True):
    """Generate mini-batches from the training data."""
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

    mini_batches = [
        data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)
    ]

    return mini_batches


def process_batch(batch, n_questions):
    question_seqs = [torch.tensor(q_ids, dtype=torch.long) for q_ids in batch['question_id']]
    correctness_seqs = [torch.tensor(correct, dtype=torch.float) for correct in batch['correct']]

    # Pad sequences to have the same length
    padded_questions = nn.utils.rnn.pad_sequence(question_seqs, batch_first=True, padding_value=0)
    padded_correctness = nn.utils.rnn.pad_sequence(correctness_seqs, batch_first=True, padding_value=-1)

    return padded_questions, padded_correctness


def train_mini_batch(rnn, data, mini_batch_size, file, model_id, init_rate):
    print('train')
    epoch_index = START_EPOCH
    optimizer = optim.Adam(rnn.parameters(), lr=init_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Retrieve the entire training dataset
    train_data = data.get_train_data()  
    # Generate mini-batches from the training data
    mini_batches = create_mini_batches(train_data, mini_batch_size, shuffle=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        rate = get_learning_rate(epoch)
        optimizer.param_groups[0]['lr'] = rate  # Update learning rate
        start_time = time.time()
        
        total_loss = 0
        predictions, labels = [], []

        # Process each mini-batch
        for batch in mini_batches:
            questions, correctness = process_batch(batch, data.n_questions)

            optimizer.zero_grad()
            outputs = rnn(questions)
            loss = criterion(outputs, correctness.view(-1, 1))

            predictions.extend(outputs.detach().sigmoid().cpu().numpy())
            labels.extend(correctness.cpu().numpy())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate metrics after processing all mini-batches
        avg_loss = total_loss / len(mini_batches)
        predictions_np = np.array(predictions).flatten()
        labels_np = np.array(labels).flatten()
        auc = roc_auc_score(labels_np, predictions_np)
        accuracy = accuracy_score(labels_np, (predictions_np > 0.5).astype(int))

        elapsed_time = time.time() - start_time
        outline = f'{epoch}\t{avg_loss}\t{auc}\t{accuracy}\t{rate}\t{elapsed_time}'
        file.write(outline + '\n')
        file.flush()
        print(outline)

        # Save the model
        torch.save(rnn.state_dict(), os.path.join(OUTPUT_DIR, 'models/', f'{model_id}_{epoch}.pth'))

        if rate == MIN_LEARNING_RATE:
            break

def get_learning_rate(epoch_index):
    rate_index = (epoch_index - 1) // LEARNING_RATE_REPEATS + 1
    rate = LEARNING_RATES[rate_index - 1] if rate_index - 1 < len(LEARNING_RATES) else MIN_LEARNING_RATE
    return rate

run('history1')