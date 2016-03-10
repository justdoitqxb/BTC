'''
Created on 2 Dec 2015
@author: hadoop
'''
import wx
import numpy as np

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.ticker import MultipleLocator, FuncFormatter
import pylab
import matplotlib.pyplot as plt
import ml.linearregresion as lr

class PlotPanel(wx.Panel):

    def __init__(self, *args, **kwargs):
        super(PlotPanel, self).__init__(*args, **kwargs)
        self.CreateComponents()

    def CreateComponents(self):
        ''' Create "interior" window components. In this case it is just a
            simple multiline text control. '''
        self.figure = matplotlib.figure.Figure()
        self.axes = self.figure.add_axes([0.1,0.1,0.8,0.8])
        self.figureCanvas = FigureCanvas(self,-1,self.figure)
        #y = np.loadtxt("../data/11214749.ECG")
        y = np.fromfile("../data/06190750.ECG", dtype=np.int32)[32:]
        #a,b = lr.nplr1(x,y)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.plot(y[0:1000])
        self.axes.grid()

        #self.control1 = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.control2 = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.sliderHOR = wx.Slider(self,100, 25, 1, 100, size=(200,-1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.sliderHOR.SetTickFreq(5,1)
        self.Bind(wx.EVT_SLIDER, self.getSliderValue, self.sliderHOR)
        self.sizerRight = wx.BoxSizer(wx.VERTICAL)
        self.sizerRight.Add(self.control2, 0, wx.EXPAND)
        self.sizerRight.Add(self.sliderHOR, 1, wx.EXEC_ASYNC)
        # Use some sizers to see layout options
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.figureCanvas, 0, wx.EXPAND)
        self.sizer.Add(self.sizerRight, 1, wx.EXPAND)

        #Layout sizers
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)

    def getSliderValue(self, event):
        value = self.sliderHOR.GetValue();
        self.control2.SetValue("%s" %value)