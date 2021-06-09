from tkinter import *
from tkinter import messagebox
import mysql.connector
import os
import time
import subprocess

#connecting to the database
db = mysql.connector.connect(host="localhost",
                            user="root",
                            passwd="paramu944",
                            database="yolo")
mycur = db.cursor()

def error_destroy():
    err.destroy()

def succ_destroy():
    succ.destroy()
    root1.destroy()

def error():
    global err
    err = Toplevel(root1)
    err.title("Error")
    err.geometry("200x100")
    Label(err,text="All fields are required..",fg="red",font="bold").pack()
    Label(err,text="").pack()
    Button(err,text="Ok",bg="grey",width=8,height=1,command=error_destroy).pack()

def success():
    global succ
    succ = Toplevel(root1)
    succ.title("Success")
    succ.geometry("200x100")
    Label(succ, text="Registration successful...", fg="green", font="bold").pack()
    Label(succ, text="").pack()
    Button(succ, text="Ok", bg="grey", width=8, height=1, command=succ_destroy).pack()

def register_user():
    username_info = username.get()
    password_info = password.get()
    if username_info == "":
        error()
    elif password_info == "":
        error()
    else:
        sql = "insert into login values(%s,%s)"
        t = (username_info, password_info)
        mycur.execute(sql, t)
        db.commit()
        Label(root1, text="").pack()
        time.sleep(0.50)
        success()



def registration():
    global root1
    root1 = Toplevel(root)
    root1.title("MOT Registration Portal")
    root1.geometry("300x250")
    global username
    global password
    Label(root1,text="Register your account",bg="grey",fg="black",font="bold",width=300).pack()
    username = StringVar()
    password = StringVar()
    Label(root1,text="").pack()
    Label(root1,text="Username :",font="bold").pack()
    Entry(root1,textvariable=username).pack()
    Label(root1, text="").pack()
    Label(root1, text="Password :").pack()
    Entry(root1, textvariable=password,show="*").pack()
    Label(root1, text="").pack()
    Button(root1,text="Register",bg="red",command=register_user).pack()

def login():
    global root2
    root2 = Toplevel(root)
    root2.title("MOT Log-In Portal")
    root2.geometry("300x300")
    global username_varify
    global password_varify
    Label(root2, text="Log-In Portal", bg="yellow", fg="black", font="bold",width=300).pack()
    username_varify = StringVar()
    password_varify = StringVar()
    Label(root2, text="").pack()
    Label(root2, text="Username :", font="bold").pack()
    Entry(root2, textvariable=username_varify).pack()
    Label(root2, text="").pack()
    Label(root2, text="Password :").pack()
    Entry(root2, textvariable=password_varify, show="*").pack()
    Label(root2, text="").pack()
    Button(root2, text="Log-In", bg="red",command=login_varify).pack()
    Label(root2, text="")

def logg_destroy():
    logg.destroy()
    root2.destroy()

def fail_destroy():
    fail.destroy()

def logged():
    global logg
    logg = Toplevel(root2)
    logg.title("MULTIPLE OBJECT TRACKING")
    logg.geometry("1000x500")
    label = Label(logg, text="Hello and welcome to MOT \n Please select your dataset !")



    def a():
        ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/yolov5s.pt --save-txt")
        if ret:
            return

    def b():
        ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/mask.pt --save-txt")
        if ret:
            return

    def c():
        ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/bedcot.pt --save-txt")
        if ret:
            return

    def d():
        ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/yolov5x.pt --save-txt")
        if ret:
            return


    def e():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5x6.pt --save-txt")
        if ret:
            return


    def f():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5m.pt --save-txt")
        if ret:
            return


    def g():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5m6.pt --save-txt")
        if ret:
            return

    def h():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5l.pt --save-txt")
        if ret:
            return

    def i():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5l6.pt --save-txt")
        if ret:
            return
    def j():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5s6.pt --save-txt")
        if ret:
            return
    def k():
        ret = subprocess.run("python yolov5/detect.py --source 0 --img 1280 --weights yolov5/weights/home.pt --save-txt")
        if ret:
            return

    def l():
        ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/watch.pt --save-txt")
        if ret:
            return

    def m():
        ret = subprocess.run("python yolov5/detect.py --source 0 --img 640 --weights yolov5/weights/penm6.pt --save-txt")
        if ret:
            return


    def n():
        ret = subprocess.run("python yolov5/detect.py --source 0 --img 640 --weights yolov5/weights/yolov5m6.pt --class 67 --save-txt")
        if ret:
            return



    b1=Button(logg,text="yolov5s dataset",bg="red",command=a)
    b2=Button(logg,text="mask dataset",command=b)
    b3=Button(logg,text="bedcot & Almirahs",command=c)
    b4=Button(logg,text="yolov5x dataset",bg="red",command=d)
    b5=Button(logg,text="yolov5x6 dataset",bg="red",command=e)
    b6=Button(logg,text="yolov5m dataset",bg="red",command=f)
    b7=Button(logg,text="yolov5m6 dataset",bg="red",command=g)
    b8=Button(logg,text="yolov5l dataset",bg="red",command=h)
    b9=Button(logg,text="yolov5l6 dataset",bg="red",command=i)
    b10=Button(logg,text="yolov5s6 dataset",bg="red",command=j)
    b11=Button(logg,text="home dataset [open cv]",command=k)
    b12=Button(logg,text="watch",command=l)
    b13=Button(logg,text="pen and pencil",command=m)
    b14=Button(logg,text="cell phone",command=n)
    b15=Button(logg, text="Log-Out", bg="grey", command=logg_destroy)

    b1.grid(row=1, column=0, padx=40, pady=15, sticky=W + N)
    b2.grid(row=2, column=0, padx=40, pady=15, sticky=W + N)
    b3.grid(row=3, column=0, padx=40, pady=15, sticky=W + N)
    b4.grid(row=1, column=1, padx=40, pady=15, sticky=W + N)
    b5.grid(row=2, column=1, padx=40, pady=15, sticky=W + N)
    b6.grid(row=3, column=1, padx=40, pady=15, sticky=W + N)
    b7.grid(row=1, column=2, padx=40, pady=15, sticky=W + N)
    b8.grid(row=2, column=2, padx=40, pady=15, sticky=W + N)
    b9.grid(row=3, column=2, padx=40, pady=15, sticky=W + N)
    b10.grid(row=1, column=3, padx=40, pady=15, sticky=W + N)
    b11.grid(row=2, column=3, padx=40, pady=15, sticky=W + N)
    b12.grid(row=3, column=3, padx=40, pady=15, sticky=W + N)
    b13.grid(row=1, column=4, padx=40, pady=15, sticky=W + N)
    b14.grid(row=2, column=4, padx=40, pady=15, sticky=W + N)
    b15.grid(row=3, column=4, padx=40, pady=15, sticky=W + N)


def failed():
    global fail
    fail = Toplevel(root2)
    fail.title("Invalid")
    fail.geometry("300x200")
    Label(fail, text="Invalid credentials...", fg="red", font="bold").pack()
    Label(fail, text="").pack()
    Button(fail, text="Ok", bg="grey", width=8, height=1, command=fail_destroy).pack()


def login_varify():
    user_varify = username_varify.get()
    pas_varify = password_varify.get()
    sql = "select * from login where user = %s and password = %s"
    mycur.execute(sql,[(user_varify),(pas_varify)])
    results = mycur.fetchall()
    if results:
        for i in results:
            logged()
            break
    else:
        failed()
def exit():
    root.destroy()

def main_screen():
    global root
    root = Tk()
    root.title("MULTIPLE OBJECT TRACKING")
    root.geometry("500x500")
    Label(root,text="Welcome to Multiple-Object-Tracking",font="bold",bg="green",fg="black",width=300).pack()
    Label(root,text="").pack()
    Button(root,text="Log-IN",width="8",height="1",bg="red",font="bold",command=login).pack()
    Label(root,text="").pack()
    Button(root, text="Registration",height="1",width="15",bg="red",font="bold",command=registration).pack()
    Label(root,text="").pack()
    Label(root,text="").pack()
    Label(root,text="Developed By Group Three ").pack()
    Label(root,text="").pack()
    Label(root,text="").pack()
    Label(root,text="").pack()
    Button(root, text="Exit",height="1",width="15",bg="blue",font="bold",command=exit).pack()

main_screen()
root.mainloop()






