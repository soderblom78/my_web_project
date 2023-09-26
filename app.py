from flask import Flask
from flask import Flask, redirect, url_for, render_template, request, flash

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")











if __name__  == "__main__":
    with app.app_context():
        app.run(debug=True)