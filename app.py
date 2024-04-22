from flask import Flask, render_template,request,jsonify

app = Flask(__name__)

KINGDOMS = [
  "Anuradhapura",
  "Polonnaruwa",
  "Dambadeniya",
  "Seethawaka",
  "Kotte",
  "Kandy",
  "Jaffna"
]

@app.route('/')
def indexPage():
  return render_template('home.html')
  
@app.route('/home')
def homePage():
  return render_template('home.html')

@app.route('/history')
def history():
  return render_template('history.html' , kingdoms = KINGDOMS)

@app.route('/inscription')
def translate():
  return render_template('inscription.html')

@app.route('/faqs')
def faqs():
  return render_template('faqs.html')

@app.route('/about')
def about():
  return render_template('about.html')
  
@app.route('/history/kingdoms', methods=["POST","GET"])
def kingdoms():
  if request.method == "POST":
      # Handle POST request
      data = request.json
      if data and 'kingdom' in data:
          kingdom = data['kingdom']
          return render_template(kingdom.lower() + '.html')
      else:
          return jsonify({'error': 'Kingdom not specified in the request data'}), 400
  else:
    
      # Handle GET request
      return render_template(kingdom)

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
