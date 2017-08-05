'''
Created on Jul 22, 2017

@author: ros
'''
import sys
from Tkinter import *
from time import sleep
import numpy as np
from collections import deque
import random
 
sys.path.append('/home/ros/eclipse-workspace/pixelhunter')

#import game_engine
from game_engine import engine
from game_engine import tFParams

def do_action(action,hunterPixelPos, x=10):
    #if action[0] == 1:
        #print("Do stop-action")
    if action[1] == 1:
        hunterPixelPos += 1
        #print("Do plus-action")
    elif action[2] == 1:
        hunterPixelPos -= 1
        #print("Do minus-action")

    if hunterPixelPos > (x - 1): 
            hunterPixelPos = (x - 1)
    elif hunterPixelPos < 0:
            hunterPixelPos = 0
            
    return hunterPixelPos

def ctrlFrameButtonSlow():
    global slow
    if slow == 0:
        slow = 1
    else:
        slow = 0
    print("slow",slow)     
    
def ctrlFrameButtonProb():
    global probability_of_random_action
    global probEntry
    
    newProb = probEntry.get()
    #print("new prob", newProb)
    probability_of_random_action = np.double(str(newProb))
    
        

if __name__ == '__main__':
    root = Tk()
    frame = Frame(root, width=150, height=150)
    frame.pack()
    app = engine.Engine(frame)
    ctrlFrame = Frame(root)
    ctrlFrame.pack( side = BOTTOM )
    slow = 0
    slowbutton = Button(ctrlFrame, text="Slow", fg="black", command=ctrlFrameButtonSlow)
    slowbutton.pack( side = BOTTOM)
    probbutton = Button(ctrlFrame, text="Prob", fg="black", command=ctrlFrameButtonProb)
    probbutton.pack(side = BOTTOM)
    probEntry = Entry(ctrlFrame)
    probEntry.pack(side = BOTTOM)
    
    
    app.createPixels()
    hunterPixelPos = 4
    app.colorHunterPixel(hunterPixelPos)
    app.colorHuntedPixel(3)
    #mainloop()
    
    tfp = tFParams.tFParams()
    
    observations = deque()
    first_run = 1
    last_state = None
    STATE_FRAMES = 4
    NUM_ACTIONS = 3
    RESIZED_DATA_X = 10 
    RESIZED_DATA_Y = 2 
    MEMORY_SIZE = 30000
    OBSERVATION_STEPS = 500
    observationStepsCounter = 0
    rewardRunStepsCounter = 0
    allRewards = 0
    
    probability_of_random_action = 0.99
    
    
    while True:
        state_from_env = app.getCurrentState()
        reward = app.getReward()
        allRewards += reward
        
        if first_run:
            first_run = 0
            last_state = np.stack(tuple(state_from_env for _ in range(STATE_FRAMES)), axis=2)
            last_action = np.zeros([NUM_ACTIONS])  #speeed of both servos 0


        state_from_env = state_from_env.reshape(RESIZED_DATA_X, RESIZED_DATA_Y, 1)
        current_state = np.append(last_state[:,:,1:], state_from_env, axis=2)
        
        observations.append((last_state, last_action, reward, current_state))
        
        if len(observations) > MEMORY_SIZE:
            observations.popleft()
            
        rewardRunStepsCounter += 1
        observationStepsCounter += 1
        if observationStepsCounter > OBSERVATION_STEPS:
            observationStepsCounter = 0
            
            tfp.setRewardAvg(allRewards)
            tfp.setRewardAvgCount(rewardRunStepsCounter)
            allRewards = 0
            rewardRunStepsCounter = 0
            
            tfp.train(observations)
            #tfp.saveSession()
        
        last_state = current_state
        last_action = tfp.choose_next_action(last_state, probability_of_random_action)
        probability_of_random_action -= 0.000001
        print(probability_of_random_action)
        
        
        if reward == 1:
            huntedPixelPos = random.randint(0, (RESIZED_DATA_X-1) )
            app.colorHuntedPixel(huntedPixelPos)
            #print("got reward")
            
            
        pos = do_action(last_action, hunterPixelPos)
        app.colorHunterPixel(pos)
        hunterPixelPos = pos
        
        if slow:
            sleep(1)