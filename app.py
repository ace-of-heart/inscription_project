from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/best_model.h5')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD1_FOLDER = 'static/uploads1'
app.config['UPLOAD1_FOLDER'] = UPLOAD1_FOLDER


KINGDOMS = [
    "Anuradhapura", "Polonnaruwa", "Dambadeniya", "Gampala", "Kotte", "Seethawaka", 
    "Kandy", "Jaffna"
]

output_letters = [
  "අ", "ආ", "ඉ", "උ", "එ", "ඔ", "ක", "කි", "කු", "කෙ", "කො", 
  "ග", "ගි", "ගු", "ගෙ", "ගො", "ච", "චි", "චු", "චෙ", "චො", 
  "ඣ", "ඣි", "ඣු", "කෙධ", "කෙධා", "ට", "ටි", "ටු", "ටෙ", "ටො", 
  "ඩ", "ඩි", "ඩු", "ඩෙ", "ඩො", "ණ", "ණි", "ණු", "ණෙ", "ණො", 
  "ත", "ති", "තු", "තෙ", "තො", "ද", "දි", "දු", "දෙ", "දො", 
  "න", "නි", "නු", "නෙ", "නො", "ප", "පි", "පු", "පෙ", "පො", 
  "බ", "බි", "බු", "බෙ", "බො", "ම", "මි", "මු", "මෙ", "මො", 
  "ය", "යි", "යු", "යෙ", "යො", "ර", "රි", "රු", "රෙ", "රො", 
  "ල", "ලි", "ලු", "ලෙ", "ලො", "ව", "වි", "වු", "වෙ", "වො", 
  "ශ", "ශි", "ශු", "ශෙ", "ශො", "ස", "සි", "සු", "සෙ", "සො", 
  "හ", "හි", "හු", "හෙ", "හො", "ඛ", "ඝ", "ඡ", "ධ", "ඵ", 
  "භ", "ළ", "ථ", "ඨ", "ජ"
  ]

label_mapping = {'1': 0, '10': 1, '100': 2, '101': 3, '102': 4, '103': 5, 
'104': 6, '105': 7, '106': 8, '107': 9, '108': 10, '109': 11, 
'11': 12, '110': 13, '111': 14, '112': 15, '113': 16, '114': 17, 
'115': 18, '12': 19, '13': 20, '14': 21, '15': 22, '16': 23, 
'17': 24, '18': 25, '19': 26, '2': 27, '20': 28, '21': 29, '22': 30, 
'23': 31, '24': 32, '25': 33, '26': 34, '27': 35, '28': 36, '29': 37,
'3': 38, '30': 39, '31': 40, '32': 41, '33': 42, '34': 43, '35': 44,
'36': 45, '37': 46, '38': 47, '39': 48, '4': 49, '40': 50, '41': 51,
'42': 52, '43': 53, '44': 54, '45': 55, '46': 56, '47': 57, '48': 58,
'49': 59, '5': 60, '50': 61, '51': 62, '52': 63, '53': 64, '54': 65,
'55': 66, '56': 67, '57': 68, '58': 69, '59': 70, '6': 71, '60': 72,
'61': 73, '62': 74, '63': 75, '64': 76, '65': 77, '66': 78, '67': 79,
'68': 80, '69': 81, '7': 82, '70': 83, '71': 84, '72': 85, '73': 86,
'74': 87, '75': 88, '76': 89, '77': 90, '78': 91, '79': 92, '8': 93,
'80': 94, '81': 95, '82': 96, '83': 97, '84': 98, '85': 99, '86': 100,
'87': 101, '88': 102, '89': 103, '9': 104, '90': 105, '91': 106,
'92': 107, '93': 108, '94': 109, '95': 110, '96': 111, '97': 112,
'98': 113, '99': 114}

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


@app.route('/upload', methods=['POST'])
def upload():
    if 'croppedImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['croppedImage']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Read the uploaded image using OpenCV
        # image = cv2.imread(filename)

      #   # Convert to grayscale
      #   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      #   # Apply thresholding to get the binary image
      #   thresh, im_bw = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
      #   # Noice Removal
      #   def noise_removal(image):
      #     kernel = np.ones((1, 1), np.uint8)
      #     image = cv2.dilate(image, kernel, iterations=1)
      #     kernel = np.ones((1, 1), np.uint8)
      #     image = cv2.erode(image, kernel, iterations=1)
      #     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
      #     image = cv2.medianBlur(image, 3)
      #     return (image)
      #   no_noise = noise_removal(im_bw)

      # # Reshape the image to fit the input shape of the model (if needed)
      #   no_noise = np.expand_dims(no_noise, axis=0)  # Add batch dimension

      # # Normalize the image (if needed)
      #   no_noise = no_noise/ 255.0

      # # Make prediction using the model
      #   prediction = model.predict(no_noise)
  


        # Invert the dictionary
        label_mapping_new = {value: key for key, value in label_mapping.items()}

          
        # Preprocess the input image
        def preprocess_image(image_path):
          img_size = 80
          img = cv2.imread(image_path)
          # plt.imshow(img)
          img = cv2.resize(img, (img_size, img_size))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img = img.reshape(1, img_size, img_size, 1)  # Convert to 4D for model prediction
          img = img / 255.0  # Normalize
          cv2.imwrite('static/uploads/processed_image.jpg',img)
          return img
          
        # Make predictions
        def predict_letter(image_path):
            input_image = preprocess_image(image_path)
            predictions = model.predict(input_image)
            predicted_letter_index = np.argmax(predictions)
            print(predicted_letter_index)
            predicted_letter = label_mapping_new[predicted_letter_index]
            output_letter = output_letters[int(predicted_letter)-1]
            return output_letter

        output_letter = predict_letter(filename)

        print(output_letter)
      
        return jsonify({'success': True, 'prediction': output_letter}), 200
      

@app.route('/upload1', methods=['POST'])
def upload1():
    if 'croppedImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['croppedImage']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD1_FOLDER'], file.filename)
        file.save(filename)


        # Invert the dictionary
        label_mapping_new = {value: key for key, value in label_mapping.items()}


        # Preprocess the input image
        def preprocess_image(image_path):
          img_size = 80
          img = cv2.imread(image_path)
          img = cv2.bitwise_not(img)
          # plt.imshow(img)
          img = cv2.resize(img, (img_size, img_size))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img = img.reshape(1, img_size, img_size, 1)  # Convert to 4D for model prediction
          img = img / 255.0  # Normalize
          cv2.imwrite('static/uploads/processed_image.jpg',img)
          return img

        # Make predictions
        def predict_letter(image_path):
            input_image = preprocess_image(image_path)
            predictions = model.predict(input_image)
            predicted_letter_index = np.argmax(predictions)
            print(predicted_letter_index)
            predicted_letter = label_mapping_new[predicted_letter_index]
            output_letter = output_letters[int(predicted_letter)-1]
            return output_letter

        output_letter = predict_letter(filename)

        print(output_letter)

        return jsonify({'success': True, 'prediction': output_letter}), 200

if __name__ == '__main__':
  if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
  if not os.path.exists(UPLOAD1_FOLDER):
    os.makedirs(UPLOAD1_FOLDER)
  app.run(host='0.0.0.0', debug=True)
