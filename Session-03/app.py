from flask import Flask, request, render_template, redirect, url_for
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence

#nltk.download('punkt_tab')

app = Flask(__name__)

# Global variables
original_lines = []
embedding_dim = 5

def preprocess_text(text: str, max_length: int = 10) -> torch.Tensor:
    """
    Preprocess a single text sentence.
    Args:
        text (str): Input text to preprocess
        max_length (int): Maximum sequence length (default: 10)
    Returns:
        torch.Tensor: Processed text as tensor with padding
    """
    # Initialize basic vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Build vocabulary from input text
    words = text.lower().split()
    for word in set(words):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Convert words to indices
    tokens = [vocab.get(word, vocab['<UNK>']) for word in text.lower().split()]
    
    # Truncate if longer than max_length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        
    # Pad if shorter than max_length
    while len(tokens) < max_length:
        tokens.append(vocab['<PAD>'])
    
    return tokens, vocab

@app.route('/', methods=['GET', 'POST'])
def index():
    global original_lines
    if request.method == 'POST':
        file = request.files['file']
        content = file.read().decode('utf-8')
        original_lines = content.splitlines()
        return redirect(url_for('choose_action'))
    return render_template('index.html')

@app.route('/choose_action')
def choose_action():
    return render_template('choose_action.html')


@app.route('/preprocess')
def preprocess():
    global original_lines
    
    combined = []
    for line in original_lines:
        # Process each line
        tokens, current_vocab = preprocess_text(line)
        
        # Convert tokens to string representation of numbers
        token_numbers = ' '.join(map(str, tokens))
        
        # Add original text and token numbers to combined list
        combined.append((line, token_numbers))
    
    return render_template('show_processed.html', combined=combined, mode='Preprocessed')

@app.route('/augment')
def augment():
    global original_lines
    augmented_lines = [augment_text(line) for line in original_lines]
    combined = zip(original_lines, augmented_lines)
    return render_template('show_processed.html', combined=combined, mode='Augmented')

def augment_text(text):
    suffix = "_aug"
    words = text.split()
    return ' '.join([word + suffix for word in words])

if __name__ == '__main__':
    app.run(debug=True, port=5001)