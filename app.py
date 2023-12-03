from flask import Flask
from flask import Flask, redirect, url_for, render_template, request, flash, send_file, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import csv
from data_preprocessing import preprocess_text, label_text
import pandas as pd


app = Flask(__name__)

UPLOAD_FOLDER = 'customer_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'xlsx'}


def allowed_file(filename):
    # Use os.path.splitext to get the file extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/")
def index():
    test_data = pd.read_csv("csv_file/test_data.csv")
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    if 'filename' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['filename']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        base_filename, extension = os.path.splitext(filename)  # Extract base filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Debug print statements
        print("File path:", file_path)
        print("Base filename:", base_filename)

        file.save(file_path)

        # Check if the CSV file contains rows with strings
        if not has_rows_with_strings(file_path, column_index=0):
            return jsonify({"error": "CSV file must contain rows with strings in the specified column"})

        # Use os.path.isfile to check if a file exists
        if os.path.isfile(file_path):
            # Load the CSV file into a Pandas DataFrame
            try:
                df = pd.read_csv(file_path)
                # Process the file
                X = df["headlines"]
                preprocessed_texts = [preprocess_text(text) for text in X]
                file_path, unique_identifier = label_text(X, preprocessed_texts, base_filename)
                return redirect(url_for('downloadpage', filename=f"{base_filename}_{unique_identifier}.csv"))

            except pd.errors.EmptyDataError:
                return jsonify({"error": "The uploaded file is empty"})

    return jsonify({"error": "Invalid file format"})



@app.route('/download/<filename>', methods=['POST'])
def download(filename):
    return send_from_directory("customer_downloads", filename, as_attachment=True)


@app.route('/downloadpage/<filename>', methods=['POST', "GET"])
def downloadpage(filename):
    return render_template("downloadpage.html", filename=filename)



def has_rows_with_strings(file_path, column_index=0):
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:  # Specify the encoding
            csv_reader = csv.reader(csvfile)
            for i, row in enumerate(csv_reader, start=1):
                if i == 1:  # Skip header row
                    continue
                
                # Check only the specified column (column_index)
                cell = row[column_index]
                
                if any(char.isalpha() for char in cell):
                    return True
            
            print(f"No rows with strings found in the specified column ({column_index}) of CSV file: {file_path}")
            return False
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    except PermissionError:
        print(f"Permission error: {file_path}")
        return False
    except Exception as e:
        print(f"Error checking CSV file: {e}")
        return False


    
    
def allowed_file(filename):
    return '.' in filename and any(filename.rsplit('.', 1)[1].lower() in ext for ext in app.config['ALLOWED_EXTENSIONS'])





if __name__  == "__main__":
    with app.app_context():
        app.run(debug=True)