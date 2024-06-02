from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

KINGDOMS = [
    "Anuradhapura", "Polonnaruwa", "Dambadeniya", "Gampala", "Kotte", "Seethawaka", 
    "Kandy", "Jaffna"
]


@app.route('/')
def indexPage():
  return render_template('home.html')


@app.route('/home')
def homePage():
  return render_template('home.html')


@app.route('/history')
def history():
  return render_template('history.html', kingdoms=KINGDOMS)


@app.route('/inscription')
def translate():
  return render_template('inscription.html')


@app.route('/faqs')
def faqs():
  return render_template('faqs.html')


@app.route('/about')
def about():
  return render_template('about.html')


@app.route('/inscription/model01')
def model01():
  return render_template('model01.html')


@app.route('/inscription/model02')
def model02():
  return render_template('model02.html')


@app.route('/inscription/model03')
def model03():
  return render_template('model03.html')


@app.route('/history/anuradhapura')
def anuradhapura():
  return render_template('anuradhapura.html')


@app.route('/history/polonnaruwa')
def polonnaruwa():
  return render_template('polonnaruwa.html')


@app.route('/history/dambadeniya')
def dambadeniya():
  return render_template('dambadeniya.html')


@app.route('/history/seethawaka')
def seethawaka():
  return render_template('seethawaka.html')


@app.route('/history/kotte')
def kotte():
  return render_template('kotte.html')


@app.route('/history/kandy')
def kandy():
  return render_template('kandy.html')


@app.route('/history/jaffna')
def jaffna():
  return render_template('jaffna.html')


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
