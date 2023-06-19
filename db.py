from flask import Flask, render_template, request ,redirect , url_for , session
from flask_session import Session

from flask_mysqldb import MySQL
from app import CF
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'parkinson'

mysql = MySQL(app)
# @app.route("/SP")
# def SP():
#     return render_template("detection.html")

# @app.route("/")
# def index():
# 	return render_template('index.html')


# @app.route("/login", methods=["POST", "GET"])
# def login():
# 	return render_template("login.html")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        details = request.form
        Patient_id = details['patient_id']
        email_id = details['email']
        # session["patient_id"] = request.form.get("patient_id")

        # paswo = details['lname']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO logindata(Patient_id, email_id) VALUES (%s, %s)", (Patient_id, email_id))
        # cur.execute("INSERT INTO MyUsers(firstName, lastName) VALUES (%s, %s)", (firstName, lastName))

        mysql.connection.commit()
        cur.close()
        return redirect(url_for('SP'))

    return render_template('login2.html')


if __name__ == '__main__':
    app.run(debug=True)

    # app.run()