from flask import Flask, request, render_template, send_from_directory, redirect, flash, url_for, session
from flask import Flask
from datetime import datetime
import random
import pandas as pd
import os
import csv_handler
import topic_modelling as tm
# from keras.models import load_model

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# app.secret_key = '%\\\xdb\xe1\x99\xec\xfb\xefU\xeb\x11Gv\xac}\x92'  # Change this!
# global reverb_model


@app.route('/')
def submit():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def submit_post():
    if request.form["action"] == 'analysis':
        option = request.form['audiofileradio']
        print(str(option))
        if option == 'csv':
            uploaded_file = request.files['audiofile']
            if uploaded_file.filename != '':
                filepath = "./files/temp/"+ uploaded_file.filename
                uploaded_file.save(filepath)
            else:
                filepath = 'unknown'

        if filepath == 'unknown':
            return render_template('home.html')

        results_bokeh = filepath
        results_csv = csv_handler.header(filepath)
        print('results')
        print(results_csv)
        print('type')
        print(type(results_csv))
        # return render_template('home_return.html', results_csv = results_csv, results_bokeh=results_bokeh, filepath=filepath)
        return redirect(url_for('topic_modelling_form', results_csv = results_csv, results_bokeh=results_bokeh, filepath=filepath))
    else:
        # filepath = request.form["filepath"]
        # timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        # new_filepath = './files/audios/audio_' + timestamp + os.path.splitext(filepath)[1]
        # os.rename(filepath,new_filepath)

        return 'thank you for submitting'



@app.route('/topic_modelling')
def topic_modelling_form():

    results_csv = request.args.get('results_csv', 'Unknown results')
    filepath = request.args.get('filepath', 'Unknown path')
    return render_template('home_return.html',results_csv = results_csv,filepath=filepath)


@app.route('/topic_modelling', methods=['POST'])
def topic_modelling_form_results():

    results_csv = request.args.get('results_csv', 'Unknown results')
    filepath = request.args.get('filepath', 'Unknown path')

    number_topics = int(request.form.get('no_topics', None))
    number_words = int(request.form.get('no_words', None))
    mode = request.form["mode"]
    print(type(number_topics))
    print (number_words)
    print (mode)
    print (filepath)
    results_topic_modelling = tm.topic_modelling_function(mode,number_topics,number_words,filepath)



    return render_template('home_return.html',results_csv = results_csv,filepath=filepath, results_topic_modelling = results_topic_modelling)


@app.route('/contribute')
def annotate():
    dirs = ['audios', 'videos']
    random_dir = random.choice(dirs)
    random_file = random.choice(os.listdir('./files/'+random_dir))
    print(random_dir)
    print(random_file)
    filepath = "./files/" + random_dir + '/' + random_file
    filepath2 = os.path.splitext(filepath)[0]
    print (filepath2)
    print (filepath)
    return render_template('contribute.html', filepath=filepath)


@app.route('/annotate', methods=["POST","GET"])
def annotate_post():
    isTampered = request.form["isTampered"]
    filepath = request.form["filepath"]
    if isTampered == 'y':
        comments = request.form["comments"]
        seconds = request.form['tampSeconds']
    else:
        comments = '-'
        seconds = '-'
    d = {'filepath': [filepath], 'isTampered': [isTampered], 'tampering point(sec)':[seconds], 'comments' : [comments]}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.dirname(os.path.abspath(filepath))+'/annotation.csv')
    return "annotation OK"


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    filename = request.args.get('filename')
    return render_template(filename)


@app.route('/files/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    files = "./files/"
    return send_from_directory(directory=files, filename=filename, as_attachment=True)

# starting point
if __name__ == '__main__':
    # serve(app)
    # global reverb_model
    # reverb_model = load_model('model_reverb.h5')
    # serve(app, host='0.0.0.0', port=8000)
    app.run()
