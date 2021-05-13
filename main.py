import subprocess
from tkinter import *
from tkinter.ttk import *


top = Tk()
top.configure(background="red")



def a():
    ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/yolov5s.pt --save-txt")
    if ret:
        return

def b():
    ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/last.pt --save-txt")
    if ret:
        return

def c():
    ret = subprocess.run("python track.py --source 0 --weights yolov5/weights/last1.pt --save-txt")
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
    ret = subprocess.run("python track.py --source 0 --img 1280 --weights yolov5/weights/yolov5m6.pt --save-txt")
    if ret:
        return

b1=Button(top,text="yolov5s dataset",command=a)
b2=Button(top,text="mask dataset",command=b)
b3=Button(top,text="bedcot & Almirahs",command=c)
b4=Button(top,text="yolov5x dataset",command=d)
b5=Button(top,text="yolov5x6 dataset",command=e)
b6=Button(top,text="yolov5m6 dataset",command=f)

b1.grid(row=1, column=0, padx=20, pady=5, sticky=W + N)
b2.grid(row=2, column=0, padx=20, pady=5, sticky=W + N)
b3.grid(row=3, column=0, padx=20, pady=5, sticky=W + N)
b4.grid(row=1, column=1, padx=20, pady=5, sticky=W + N)
b5.grid(row=2, column=1, padx=20, pady=5, sticky=W + N)
b6.grid(row=3, column=1, padx=20, pady=5, sticky=W + N)


mainloop() 