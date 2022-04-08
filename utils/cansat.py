import sys
from tracemalloc import start
import pandas as pd
import matplotlib.pyplot as plt 

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import QtWidgets, QtCore
import threading
import time
import collections
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import numpy as np 
from serial import Serial

import time

from oscuro_grafica import *

serialPort='COM20'
baudRate=9600

class Worker(QtCore.QObject):

    lineLabel ='Valores'
    Samples = 20 
    data=collections.deque([0]*Samples, maxlen=Samples)
    data1=collections.deque([0]*Samples, maxlen=Samples)
    fig, axes = plt.subplots(1,2)
    lines=axes[0].plot([],[],'tab:red',label=lineLabel,linewidth=2)[0]
    lines1=axes[0].plot([],[],'tab:blue',label=lineLabel,linewidth=2)[0]
    serialConnection = Serial(serialPort, baudRate)
    finished = pyqtSignal()
    datos_sensor = pyqtSignal()
    
    def getSerialData(self):
    
        print('------------------') 
        
        for i in range(5):
            time.sleep(5)
            self.datos_sensor.emit(i)
            print(i)
        """value=self.serialConnection.readline().strip()
        print(value)
        value=value.decode("utf-8")
        value=value.split('&')
        for i in range(len(value)):
            value[i]=float(value[i])

        print(value)   
        self.data.append(value[0])
        self.lines.set_data(range(self.Samples),self.data)
        self.data1.append(value[1])
        self.lines1.set_data(range(self.Samples),self.data1)



        serialPort='COM20'
        baudRate=9600



        try:
            self.serialConnection = Serial(serialPort, baudRate)
        except:
            print("No se realizó conexion")
            

        Samples = 20 
        self.data=collections.deque([0]*self.Samples, maxlen=self.Samples)
        self.data1=collections.deque([0]*self.Samples, maxlen=self.Samples)
        sampleTime=100

        xmin=0
        xmax=Samples
        ymin=0
        ymax=6



        lineLabel ='Valores'
        #fig, axes = plt.subplots(1,2, figsize=(10, 10))
        fig, axes = plt.subplots(1,2)

        axes[0]=fig.add_subplot(1,2,1,xlim=(xmin,20),ylim=(ymin,ymax))
        axes[0].title.set_text('Primera gráfica')
        axes[0].set_xlabel("Muestras")
        axes[0].set_ylabel("Valores")
        self.lines=axes[0].plot([],[],'tab:red',label=lineLabel,linewidth=2)[0]
        axes[0].set_ylim(0, 6)


        ax2=fig.add_subplot(1,2,2,xlim=(xmin,20),ylim=(ymin,ymax))
        ax2.title.set_text('Segunda gráfica')
        ax2.set_xlabel("Muestras")
        ax2.set_ylabel("Valores")
        self.lines1=ax2.plot([],[],label=lineLabel,linewidth=2)[0]
        plt.axis('off')


        #anim=animation.FuncAnimation(fig,self.getSerialData,interval=sampleTime)
        #plt.show()
        #fig.savefig('imagen2.png', transparent=True)
        #plt.close(self.fig)sam

        self.serialConnection.close()"""
        self.finished.emit()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form() 
        self.ui.setupUi(self)
        self.setWindowTitle("Laika")
        self.setWindowIcon(QIcon("./sources/layca_image.jpg"))
        self.viewTemperature()
        #self.on()
        
        self.timer = QtCore.QTimer()
        self.timer.start(5000)
        self.timer.timeout.connect(self.on)
        #self.timer.timeout.connect(self.viewTemperature)
    

    def viewTemperature(self,datos_sensor):
        try:
            datos = (datos_sensor,datos_sensor,datos_sensor,datos_sensor,datos_sensor)
            index_datos = (0,1,2,3,4,5)

            df=pd.DataFrame(datos, index_datos) 
            ax = df.plot()
            fig = ax.get_figure()
            plt.tick_params(axis='x', labelsize=18)
            plt.tick_params(axis='y', labelsize=18)

            # set various colors
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white', which='both')  # 'both' refers to minor and major axes

            fig = ax.get_figure()
                
            fig.savefig('imagen.png',transparent=True)
            plt.close(fig)
        
            pixmap = QPixmap('imagen.png')
            #Actualizamos la imagen
            self.ui.LIVE_14.setPixmap(pixmap) 
        except:
            pass
        
    def on(self):
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.getSerialData)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.datos_sensor.connect(self.viewTemperature)
        self.thread.finished.connect(self.thread.deleteLater)
        # Step 6: Start the thread
        self.thread.start()
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()



        