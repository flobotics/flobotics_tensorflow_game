'''
Created on Jul 22, 2017

@author: ros
'''
from Tkinter import Frame
from Tkinter import Canvas
from Tkinter import NW 
from Tkinter import Tk
import numpy as np
from PIL import Image, ImageTk

class Engine(Frame):
    '''
    classdocs
    '''
    def colorHunterPixel(self,nr):
        if self.x != nr:
            self.canvas.create_rectangle(10+(nr*10),10,20+(nr*10),20,fill="blue")
            self.canvas.create_rectangle(10+(self.x*10),10,20+(self.x*10),20,fill="grey")
            self.x = nr
            self.root.update()
        
    def colorHuntedPixel(self,nr):
        if self.y != nr:
            self.canvas.create_rectangle(10+(nr*10),20,20+(nr*10),30,fill="red")
            self.canvas.create_rectangle(10+(self.y*10),20,20+(self.y*10),30,fill="grey")
            self.y = nr
            self.root.update()
        
    def createPixels(self, countPixelX=10, countPixelY=2):
        for j in range(countPixelY):
            for i in range(countPixelX):
                self.canvas.create_rectangle(10+(i*10),10+(j*10),20+(i*10),20+(j*10),fill="grey")   
        
        self.maxX = countPixelX
        self.maxY = countPixelY
        self.root.update()
        
    def getCurrentState(self):
        a = np.zeros([self.maxX,self.maxY])
        a[self.x][0] = 1
        a[self.y][1] = 1
        return a
        
    def getReward(self):
        if self.x == self.y:
            self.lastDiff = 0
            return 1
        elif self.x > self.y:
            diff = self.x - self.y
            preDiff = self.lastDiff
            self.lastDiff = diff
                     
        elif self.x < self.y:
            diff = self.y - self.x
            preDiff = self.lastDiff
            self.lastDiff = diff
            
        if preDiff > diff: 
            return 0.1
        elif preDiff < diff:
            return -1
        elif preDiff == diff:
            return -0.1

    def __init__(self, master):
        '''
        Constructor
        '''
        self.x = 0
        self.y = 0
        self.maxX = 0
        self.maxY = 0
        self.lastDiff = 0
        self.root = master
        self.canvas = Canvas(self.root, width=150,height=150)
        self.canvas.pack()
        #self.createWidgets()
        #self.root.mainloop()