# from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


from flask_session import Session

from flask_mysqldb import MySQL
from flask import Flask, render_template, request, redirect, url_for ,session
import os
import pickle
import pandas as pd
import numpy as np
from os.path import join, dirname, realpath
from sklearn.preprocessing import StandardScaler
from math import sqrt


app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'parkinson'

mysql = MySQL(app)
# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('models\model_v.pkl', 'rb'))
scaler = StandardScaler()

model_SP = pickle.load(open('models\model_SP.pkl', 'rb'))

@app.route("/Compute_cf")
def Compute_cf():
    cur = mysql.connection.cursor()
    cur1 = mysql.connection.cursor()
    cur2 = mysql.connection.cursor() #motor
    id = (session["patient_id"])
    print(id)
    cur.execute("select SnP_score from snp where patient_id = %s", (session["patient_id"]))
    cur1.execute("select v_score from v_impairment where patient_id = %s", (session["patient_id"]))
    cur2.execute("select m_score from motor where patient_id = %s", (session["patient_id"])) # motor

    data2 = cur.fetchall()
    print(data2)

    data2 = cur.fetchall()
    print(data2)
    data1 = cur1.fetchall()
    print(data1)
    s_score = data2[0][0]
    v_score = data1[0][0]

    avg = float((s_score + v_score)/2)

    cur3 = mysql.connection.cursor()
    cur3.execute("INSERT INTO cfs(Patient_id, CFS_score) VALUES (%s, %s)", (session["patient_id"], avg))
    mysql.connection.commit()
    cur3.close()

    cur4 = mysql.connection.cursor()
    cur4.execute("SELECT snp.SnP_score, v_impairment.v_score, cfs.CFS_score FROM snp JOIN v_impairment ON v_impairment.patient_id = snp.patient_id JOIN cfs ON cfs.patient_id = snp.patient_id where snp.patient_id = %s" , (session["patient_id"]) )
    data = cur4.fetchall()



    # print("fetched from db")
    print(s_score)
    print(data)
    return render_template('CF.html', data=data)

@app.route("/logout")
def logout():
	session["patient_id"] = None
	return redirect("/")


@app.route("/login", methods=["POST", "GET"])
def login():
  # if form is submited
        if request.method == "POST":
            details = request.form
            Patient_id = details['patient_id']
            email_id = details['email']
            session["patient_id"] = request.form.get("patient_id")
            print("omkar")
            print(session["patient_id"])
            # paswo = details['lname']
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO logindata(Patient_id, email_id) VALUES (%s, %s)", (Patient_id, email_id))
            # cur.execute("INSERT INTO MyUsers(firstName, lastName) VALUES (%s, %s)", (firstName, lastName))
            mysql.connection.commit()
            cur.close()
            # return redirect(url_for('SP'))
            return redirect("/CF_SP")

        return render_template("login2.html")   

@app.route('/', methods=['GET'])
def index():
    
    if not session.get("patient_id"):
        # if not there in the session then redirect to the login page
        return redirect("/login")
    return render_template('index.html')


@app.route('/index', methods=['GET'])
def AfterRegister():
    # Main page
    return render_template('index.html')


@app.route("/SP")
def SP():
    return render_template("detection.html")

@app.route("/CF_SP")
def CF_SP():
    return render_template("CF_SP.html")

@app.route("/CF")
def CF():
    return render_template("CF.html")

@app.route("/CF_VI")
def CF_VI():
    return render_template("CF_VI.html")

# ----------------------------SP FUNCTIONS------------------------------
def get_no_strokes(df):
    pressure_data=df['Pressure'].to_numpy()
    on_surface = (pressure_data>600).astype(int)
    return ((np.roll(on_surface, 1) - on_surface) != 0).astype(int).sum()

def get_speed(df):
    total_dist=0
    duration=df['Timestamp'].to_numpy()[-1]
    coords=df[['X', 'Y', 'Z']].to_numpy()
    for i in range(10, df.shape[0]):
        temp=np.linalg.norm(coords[i, :]-coords[i-10, :])
        total_dist+=temp
    speed=total_dist/duration
    return speed

def get_in_air_time(data):
    data=data['Pressure'].to_numpy()
    return (data<600).astype(int).sum()

def get_on_surface_time(data):
    data=data['Pressure'].to_numpy()
    return (data>600).astype(int).sum()

def find_velocity(f):

    data_pat=f
    Vel = []
    horz_Vel = []
    horz_vel_mag = []
    vert_vel_mag = []
    vert_Vel = []
    magnitude = []
    timestamp_diff =  []
    
    t = 0
    for i in range(len(data_pat)-2):
        if t+10 <= len(data_pat)-1:
            Vel.append(((data_pat['X'].to_numpy()[t+10] - data_pat['X'].to_numpy()[t])/(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t]) , (data_pat['Y'].to_numpy()[t+10]-data_pat['Y'].to_numpy()[t])/(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t])))
            horz_Vel.append((data_pat['X'].to_numpy()[t+10] - data_pat['X'].to_numpy()[t])/(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t]))
            
            vert_Vel.append((data_pat['Y'].to_numpy()[t+10] - data_pat['Y'].to_numpy()[t])/(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t]))
            magnitude.append(sqrt(((data_pat['X'].to_numpy()[t+10]-data_pat['X'].to_numpy()[t])/(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t]))**2 + (((data_pat['Y'].to_numpy()[t+10]-data_pat['Y'].to_numpy()[t])/(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t]))**2)))
            timestamp_diff.append(data_pat['Timestamp'].to_numpy()[t+10]-data_pat['Timestamp'].to_numpy()[t])
            horz_vel_mag.append(abs(horz_Vel[len(horz_Vel)-1]))
            vert_vel_mag.append(abs(vert_Vel[len(vert_Vel)-1]))
            t = t+10
        else:
            break
    magnitude_vel = np.mean(magnitude)  
    magnitude_horz_vel = np.mean(horz_vel_mag)
    magnitude_vert_vel = np.mean(vert_vel_mag)
    return Vel,magnitude,timestamp_diff,horz_Vel,vert_Vel,magnitude_vel,magnitude_horz_vel,magnitude_vert_vel

def find_acceleration(f):
    Vel,magnitude,timestamp_diff,horz_Vel,vert_Vel,magnitude_vel,magnitude_horz_vel,magnitude_vert_vel = find_velocity(f)
    accl = []
    horz_Accl =  []
    vert_Accl = []
    magnitude = []
    horz_acc_mag = []
    vert_acc_mag = []
    for i in range(len(Vel)-2):
        accl.append(((Vel[i+1][0]-Vel[i][0])/timestamp_diff[i] , (Vel[i+1][1]-Vel[i][1])/timestamp_diff[i]))
        horz_Accl.append((horz_Vel[i+1]-horz_Vel[i])/timestamp_diff[i])
        vert_Accl.append((vert_Vel[i+1]-vert_Vel[i])/timestamp_diff[i])
        horz_acc_mag.append(abs(horz_Accl[len(horz_Accl)-1]))
        vert_acc_mag.append(abs(vert_Accl[len(vert_Accl)-1]))
        magnitude.append(sqrt(((Vel[i+1][0]-Vel[i][0])/timestamp_diff[i])**2 + ((Vel[i+1][1]-Vel[i][1])/timestamp_diff[i])**2))
    
    magnitude_acc = np.mean(magnitude)  
    magnitude_horz_acc = np.mean(horz_acc_mag)
    magnitude_vert_acc = np.mean(vert_acc_mag)
    return accl,magnitude,horz_Accl,vert_Accl,timestamp_diff,magnitude_acc,magnitude_horz_acc,magnitude_vert_acc
    

def find_jerk(f):
    accl,magnitude,horz_Accl,vert_Accl,timestamp_diff,magnitude_acc,magnitude_horz_acc,magnitude_vert_acc = find_acceleration(f)
    jerk = []
    hrz_jerk = []
    vert_jerk = []
    magnitude = []
    horz_jerk_mag = []
    vert_jerk_mag = []
    
    for i in range(len(accl)-2):
        jerk.append(((accl[i+1][0]-accl[i][0])/timestamp_diff[i] , (accl[i+1][1]-accl[i][1])/timestamp_diff[i]))
        hrz_jerk.append((horz_Accl[i+1]-horz_Accl[i])/timestamp_diff[i])
        vert_jerk.append((vert_Accl[i+1]-vert_Accl[i])/timestamp_diff[i])
        horz_jerk_mag.append(abs(hrz_jerk[len(hrz_jerk)-1]))
        vert_jerk_mag.append(abs(vert_jerk[len(vert_jerk)-1]))
        magnitude.append(sqrt(((accl[i+1][0]-accl[i][0])/timestamp_diff[i])**2 + ((accl[i+1][1]-accl[i][1])/timestamp_diff[i])**2))
        
    magnitude_jerk = np.mean(magnitude)  
    magnitude_horz_jerk = np.mean(horz_jerk_mag)
    magnitude_vert_jerk = np.mean(vert_jerk_mag)
    return jerk,magnitude,hrz_jerk,vert_jerk,timestamp_diff,magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk


def NCV_per_halfcircle(f):
    data_pat=f
    Vel = []
    ncv = []
    temp_ncv = 0
    basex = data_pat['X'].to_numpy()[0]
    for i in range(len(data_pat)-2):
        if data_pat['X'].to_numpy()[i] == basex:
            ncv.append(temp_ncv)
            temp_ncv = 0
            continue
            
        Vel.append(((data_pat['X'].to_numpy()[i+1] - data_pat['X'].to_numpy()[i])/(data_pat['Timestamp'].to_numpy()[i+1]-data_pat['Timestamp'].to_numpy()[i]) , (data_pat['Y'].to_numpy()[i+1]-data_pat['Y'].to_numpy()[i])/(data_pat['Timestamp'].to_numpy()[i+1]-data_pat['Timestamp'].to_numpy()[i])))
        if Vel[len(Vel)-1] != (0,0):
            temp_ncv+=1
    ncv.append(temp_ncv)
    ncv_Val = np.sum(ncv)/np.count_nonzero(ncv)
    return ncv,ncv_Val

def NCA_per_halfcircle(f):
    data_pat=f
    Vel,magnitude,timestamp_diff,horz_Vel,vert_Vel,magnitude_vel,magnitude_horz_vel,magnitude_vert_vel = find_velocity(f)
    accl = []
    nca = []
    temp_nca = 0
    basex = data_pat['X'].to_numpy()[0]
    for i in range(len(Vel)-2):
        if data_pat['X'].to_numpy()[i] == basex:
            nca.append(temp_nca)
            #print ('tempNCa::',temp_nca)
            temp_nca = 0
            continue
            
        accl.append(((Vel[i+1][0]-Vel[i][0])/timestamp_diff[i] , (Vel[i+1][1]-Vel[i][1])/timestamp_diff[i]))
        if accl[len(accl)-1] != (0,0):
            temp_nca+=1
    nca.append(temp_nca)
    nca = list(filter((2).__ne__, nca))
    nca_Val = np.sum(nca)/np.count_nonzero(nca)
    return nca,nca_Val

def get_features_test(f):
    global header_row
    df=pd.read_csv(f, sep=';', header=None, names=header_row)
    
    df_static=df[df["Test_ID"]==0]    # static test
    df_dynamic=df[df["Test_ID"]==1]    # dynamic test
    df_stcp=df[df["Test_ID"]==2]    # STCP
    #df_static_dynamic=pd.concat([df_static, df_dynamic])
    
    initial_timestamp=df['Timestamp'][0]
    df['Timestamp']=df['Timestamp']- initial_timestamp # offset timestamps
    
    duration_static = df_static['Timestamp'].to_numpy()[-1] if df_static.shape[0] else 1
    duration_dynamic = df_dynamic['Timestamp'].to_numpy()[-1] if df_dynamic.shape[0] else 1
    duration_STCP = df_stcp['Timestamp'].to_numpy()[-1] if df_stcp.shape[0] else 1

    
    data_point=[]
    data_point.append(get_no_strokes(df_static) if df_static.shape[0] else 0) # no. of strokes for static test
    data_point.append(get_no_strokes(df_dynamic) if df_dynamic.shape[0] else 0) # no. of strokes for dynamic test
    data_point.append(get_speed(df_static) if df_static.shape[0] else 0) # speed for static test
    data_point.append(get_speed(df_dynamic) if df_dynamic.shape[0] else 0) # speed for dynamic test

    Vel,magnitude,timestamp_diff,horz_Vel,vert_Vel,magnitude_vel,magnitude_horz_vel,magnitude_vert_vel = find_velocity(df_static) if df_static.shape[0] else (0,0,0,0,0,0,0,0) # magnitudes of velocity, horizontal velocity and vertical velocity for static test
    data_point.extend([magnitude_vel, magnitude_horz_vel,magnitude_vert_vel])
    Vel,magnitude,timestamp_diff,horz_Vel,vert_Vel,magnitude_vel,magnitude_horz_vel,magnitude_vert_vel = find_velocity(df_dynamic) if df_dynamic.shape[0] else (0,0,0,0,0,0,0,0) # magnitudes of velocity, horizontal velocity and vertical velocity for dynamic test
    data_point.extend([magnitude_vel, magnitude_horz_vel,magnitude_vert_vel])
    
    accl,magnitude,horz_Accl,vert_Accl,timestamp_diff,magnitude_acc,magnitude_horz_acc,magnitude_vert_acc=find_acceleration(df_static) if df_static.shape[0] else (0,0,0,0,0,0,0,0)# magnitudes of acceleration, horizontal acceleration and vertical acceleration for static test        
    data_point.extend([magnitude_acc,magnitude_horz_acc,magnitude_vert_acc])
    accl,magnitude,horz_Accl,vert_Accl,timestamp_diff,magnitude_acc,magnitude_horz_acc,magnitude_vert_acc=find_acceleration(df_dynamic) if df_dynamic.shape[0] else (0,0,0,0,0,0,0,0)# magnitudes of acceleration, horizontal acceleration and vertical acceleration for dynamic test        
    data_point.extend([magnitude_acc,magnitude_horz_acc,magnitude_vert_acc])
    
    jerk,magnitude,hrz_jerk,vert_jerk,timestamp_diff,magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk=find_jerk(df_static) if df_static.shape[0] else (0,0,0,0,0,0,0,0) # magnitudes of jerk, horizontal jerk and vertical jerk for static test
    data_point.extend([magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk])
    jerk,magnitude,hrz_jerk,vert_jerk,timestamp_diff,magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk=find_jerk(df_dynamic) if df_dynamic.shape[0] else (0,0,0,0,0,0,0,0) # magnitudes of jerk, horizontal jerk and vertical jerk for dynamic test
    data_point.extend([magnitude_jerk,magnitude_horz_jerk,magnitude_vert_jerk])
    
    ncv,ncv_Val=NCV_per_halfcircle(df_static) if df_static.shape[0] else (0,0) # NCV for static test
    data_point.append(ncv_Val)
    ncv,ncv_Val=NCV_per_halfcircle(df_dynamic) if df_dynamic.shape[0] else (0,0) # NCV for dynamic test
    data_point.append(ncv_Val)
    
    nca,nca_Val=NCA_per_halfcircle(df_static) if df_static.shape[0] else (0,0) # NCA for static test
    data_point.append(nca_Val)
    nca,nca_Val=NCA_per_halfcircle(df_dynamic) if df_dynamic.shape[0] else (0,0) # NCA for dynamic test
    data_point.append(nca_Val)
    
    data_point.append(get_in_air_time(df_stcp) if df_stcp.shape[0] else 0) # in air time for STCP
    data_point.append(get_on_surface_time(df_static) if df_static.shape[0] else 0) # on surface time for static test
    data_point.append(get_on_surface_time(df_dynamic) if df_dynamic.shape[0] else 0) # on surface time for dynamic test
    
    # data_point.append(parkinson_target)    # traget. 1 for parkinson. 0 for control.
    
    return data_point

header_row=["X", "Y", "Z", "Pressure" , "GripAngle" , "Timestamp" , "Test_ID"]

@app.route('/predict', methods=['GET','POST'])
def uploadFiles():
    # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            raw=[]
            raw.append(get_features_test(file_path))
            raw=np.array(raw)
            features_headers=['no_strokes_st', 'no_strokes_dy', 'speed_st', 'speed_dy', 'magnitude_vel_st' , 'magnitude_horz_vel_st' , 'magnitude_vert_vel_st', 'magnitude_vel_dy', 'magnitude_horz_vel_dy' , 'magnitude_vert_vel_dy', 'magnitude_acc_st' , 'magnitude_horz_acc_st' , 'magnitude_vert_acc_st','magnitude_acc_dy' , 'magnitude_horz_acc_dy' , 'magnitude_vert_acc_dy', 'magnitude_jerk_st', 'magnitude_horz_jerk_st' , 'magnitude_vert_jerk_st', 'magnitude_jerk_dy', 'magnitude_horz_jerk_dy' , 'magnitude_vert_jerk_dy', 'ncv_st', 'ncv_dy', 'nca_st', 'nca_dy', 'in_air_stcp','on_surface_st', 'on_surface_dy']
            data_test=pd.DataFrame(raw, columns=features_headers)

            prediction = model_SP.predict(data_test.iloc[0:1])
            class_probabilities = model_SP.decision_function(data_test.iloc[0:1])
            print(session["patient_id"])
            # paswo = details['lname']
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO snp(Patient_id, SnP_score) VALUES (%s, %s)", (session["patient_id"], class_probabilities[0]))
            mysql.connection.commit()
            cur.close()
            print("This awdadawd")
            print(class_probabilities)

            if prediction[0] == 1:
                predictiona = 'Parkinson disease positive'
            else:
                predictiona = 'Normal'
            return predictiona
    return None


@app.route("/VI")
def VI():
    return render_template("detect.html")

# visual impairments
@app.route('/prediction', methods=['GET','POST'])
def uploadFiless():
    # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            # save the file
            r_data = pd.read_csv(file_path)
            r_data_np = np.asarray(r_data)
            r_data_shape = r_data_np.reshape(1, -1)
            real_world_data = scaler.fit_transform(r_data_shape)
            prediction = model.predict(real_world_data)
            class_probabilities = model.decision_function(real_world_data)
            print(session["patient_id"])
            # paswo = details['lname']
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO v_impairment(Patient_id, v_score) VALUES (%s, %s)", (session["patient_id"], class_probabilities[0]))
            mysql.connection.commit()
            cur.close()
            print(prediction)
            # class_probabilities = model.predict_proba(real_world_data)
            # print("This awdadawd")
            # print(class_probabilities)

            if prediction[0] == 1:
                predictiona = 'Parkinson disease positive'
            else:
                predictiona = 'Normal'
            return predictiona
    return None

@app.route('/singlepredict', methods=['GET','POST'])
def singleuploadFiles():
    # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            raw=[]
            raw.append(get_features_test(file_path))
            raw=np.array(raw)
            features_headers=['no_strokes_st', 'no_strokes_dy', 'speed_st', 'speed_dy', 'magnitude_vel_st' , 'magnitude_horz_vel_st' , 'magnitude_vert_vel_st', 'magnitude_vel_dy', 'magnitude_horz_vel_dy' , 'magnitude_vert_vel_dy', 'magnitude_acc_st' , 'magnitude_horz_acc_st' , 'magnitude_vert_acc_st','magnitude_acc_dy' , 'magnitude_horz_acc_dy' , 'magnitude_vert_acc_dy', 'magnitude_jerk_st', 'magnitude_horz_jerk_st' , 'magnitude_vert_jerk_st', 'magnitude_jerk_dy', 'magnitude_horz_jerk_dy' , 'magnitude_vert_jerk_dy', 'ncv_st', 'ncv_dy', 'nca_st', 'nca_dy', 'in_air_stcp','on_surface_st', 'on_surface_dy']
            data_test=pd.DataFrame(raw, columns=features_headers)

            prediction = model_SP.predict(data_test.iloc[0:1])
            print(prediction)

            if prediction[0] == 1:
                predictiona = 'Parkinson disease positive'
            else:
                predictiona = 'Normal'
            return predictiona
    return None



@app.route('/singleprediction', methods=['GET','POST'])
def singleuploadFiless():
    # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            # save the file
            r_data = pd.read_csv(file_path)
            r_data_np = np.asarray(r_data)
            r_data_shape = r_data_np.reshape(1, -1)
            real_world_data = scaler.fit_transform(r_data_shape)
            prediction = model.predict(real_world_data)
            print(prediction)

            if prediction[0] == 1:
                predictiona = 'Parkinson disease positive'
            else:
                predictiona = 'Normal'
            return predictiona
    return None


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='192.168.0.106')
