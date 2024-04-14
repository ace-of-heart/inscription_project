from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def indexPage():
  return render_template('home.html')
  
@app.route('/home')
def homePage():
  return render_template('home.html')

@app.route('/history')
def history():
  return render_template('history.html')

@app.route('/inscription')
def translate():
  return render_template('inscription.html')

@app.route('/faqs')
def faqs():
  return render_template('faqs.html')

@app.route('/about')
def about():
  return render_template('about.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
