# coding:utf-8
import tkinter as tk
from myEBPTA import *


class GUIofCR:
    plate = []
    trainset = []
    bptool = myEBPTA(25, 50, 26) # (sample_length, output_length, hidden_length)，change these numbers to adjust the model

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Character Recognition.")
        root = self.root

        # root.geometry('400x600')

        for i in range(25):
            c = myCanvas(i, root)
            self.plate.append(c)

        self.label1 = tk.Label(root, text="绘图区 (made by liguanyu)")
        self.label1.grid(row=5, columnspan=5)
        self.trainbutton = tk.Button(root, text="训练", command=self.train, width=15, height=2)
        self.trainbutton.grid(row=6, column=1, columnspan=3)
        self.recogbutton = tk.Button(root, text="识别", command=self.recog, width=15, height=2)
        self.recogbutton.grid(row=7, column=1, columnspan=3)
        self.label1 = tk.Label(root, text="检测结果：")
        self.label1.grid(row=8, rowspan=3, column=0, columnspan=2)

        self.resultstr = tk.StringVar()
        self.resultentry = tk.Entry(root, textvariable=self.resultstr)
        self.resultentry.grid(row=8, rowspan=3, column=2, columnspan=3)
        self.resultentry['state'] = 'readonly'

        root.mainloop()

    def train(self):
        file = open("trainset.txt", 'r')
        content = file.readlines()
        for line in content:
            data = line.split()
            self.trainset.append(ord(data[0]) - 65 + 1)
            self.trainset.append(np.array(list(map(int, data[1:]))))
        for i in range(len(self.trainset)):
            if i % 2 == 0:
                self.bptool.addtrainset(self.trainset[i + 1], self.trainset[i])
        self.bptool.initProcess()
        self.bptool.start()

    def recog(self):
        sample = []
        for i in range(25):
            sample.append(self.plate[i].selected)
        sample = np.array(sample)
        classnum = self.bptool.recognize(sample)
        if classnum == 0:
            self.resultstr.set("未能成功匹配")
        else:
            self.resultstr.set("识别为'%s'" % (chr(classnum + 64)))


class myCanvas:
    index = 0
    selected = 0

    def __init__(self, index, root):
        self.index = index
        self.plate = tk.Canvas(root, width=40, height=40, bg='yellow', cursor="dot")
        self.plate.bind("<Button-1>", self.press)
        self.plate.grid(row=(index // 5), column=(index % 5))

    def press(self, event):
        if self.selected == 0:
            self.selected = 1
            self.plate['bg'] = 'black'
        elif self.selected == 1:
            self.selected = 0
            self.plate['bg'] = 'yellow'
