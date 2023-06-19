from flask import Flask, render_template, redirect, request, session


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
def index():
	return render_template('index.html')


@app.route("/login", methods=["POST", "GET"])
def login():
	return render_template('login2.html")

@app.route("/login", methods=["POST", "GET"])
def login():
# if form is submited
	if request.method == "POST":
		# record the user name
		session["name"] = request.form.get("name")
		# redirect to the main page
		return redirect("/")
	return render_template("login.html")
    
