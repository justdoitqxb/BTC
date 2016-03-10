'''
Created on 2 Dec 2015
@author: hadoop
'''
import wx
import os.path

class FilePanel(wx.Panel):
    
    def __init__(self, *args, **kwargs):
        super(FilePanel, self).__init__(*args, **kwargs)
        self.filename = 'None'
        self.dirname = '.'
        self.CreateComponents()
       # self.CreateSplitter()

    def CreateComponents(self):
        ''' Create "interior" window components. In this case it is just a
            simple multiline text control. '''
        
        self.control1 = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.control2 = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.sliderHOR = wx.Slider(self,100, 25, 1, 100, size=(200,-1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.sliderHOR.SetTickFreq(5,1)
        self.Bind(wx.EVT_SLIDER, self.getSliderValue, self.sliderHOR)
        self.sizerRight = wx.BoxSizer(wx.VERTICAL)
        self.sizerRight.Add(self.control2, 0, wx.EXPAND)
        self.sizerRight.Add(self.sliderHOR, 1, wx.EXEC_ASYNC)
        # Use some sizers to see layout options
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.control1, 0, wx.EXPAND)
        self.sizer.Add(self.sizerRight, 1, wx.EXPAND)
 
        #Layout sizers
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        
    def CreateSplitter(self):
        # Create the splitter window.
        splitter = wx.SplitterWindow(self, style=wx.NO_3D|wx.SP_3D)
        control1 = wx.TextCtrl(splitter, style=wx.TE_MULTILINE)
        control2 = wx.TextCtrl(splitter, style=wx.TE_MULTILINE)
        splitter.SetMinimumPaneSize(1)
        # Install the tree and the editor.
        splitter.SplitVertically(self.control1, self.control2)
        splitter.SetSashPosition(580, True)
        #self.Bind(wx.EVT_CLOSE, self.OnProjectExit)
        #self.Show(True)
        
    def getSliderValue(self, event):
        value = self.sliderHOR.GetValue();
        self.control2.SetValue("%s" %value) 