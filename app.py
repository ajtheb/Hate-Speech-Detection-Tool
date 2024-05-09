from flask import Flask, render_template, request, current_app
from helpers import model_load, inference, tokenizer_load
app = Flask(__name__)
app.config['MODEL'] = model_load()
app.config['TOKENIZER'] = tokenizer_load()

@app.route('/')
def index(): 
    print("Model loaded successfully!!")
    return render_template('index.html')

@app.route('/highlight', methods=['POST'])
def highlight():
    paragraph = request.values.get('paragraph')
    print("Inside highlight!")
    print(paragraph)
    sentences,sentences_probs  = inference(paragraph,current_app.config['MODEL'],current_app.config['TOKENIZER'])
    
    highlighted_indices = []
    c = 0 
    for p in sentences_probs:
        if(p>=0.5):
            highlighted_indices.append(c)
        c+=1
    sentences_probs*=100

    sentences_probs = [int(x) for x in sentences_probs]


    sentence_packet = [[x,y,z] for x,y,z in zip(range(len(sentences)),sentences,sentences_probs)]
    # Your highlighting logic here
    return render_template('index.html', paragraph=paragraph, highlighted_indices=highlighted_indices, sentences = sentence_packet)

if __name__ == '__main__':
    app.run(debug=True)
