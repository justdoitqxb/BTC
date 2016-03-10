# -*- coding: gbk -*-
'''
Created on 2 Dec 2015
@author: hadoop
'''
import wx
import wx.py.images
import os.path
from drawpanel import *
from plotpanel import *
from form import *
from filepanel import *

class MainWindow(wx.Frame):
    def __init__(self, filename='none'):
        super(MainWindow, self).__init__(None, size=(800,600))
        self.filename = filename
        self.dirname = '.'
        self.SetBackgroundColour('gred')
        self.CreateWindowComponents()
        noteBook = wx.Notebook(self)
        noteBook.AddPage(FilePanel(noteBook), '文件列表')
        noteBook.AddPage(PlotPanel(noteBook), '心电图分析')
        noteBook.AddPage(FormWithAbsolutePositioning(noteBook), '综合分析')
        noteBook.AddPage(DoodlePanel(noteBook), '心情绘图')
        noteBook.AdvanceSelection(2)

    def CreateWindowComponents(self):
        ''' Create window components, such as menu and status
            bar. '''
        self.CreateMenuBar()
        self.CreateStatusAndTool()
        self.SetTitle()
        
    def CreateMenuBar(self):
        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()
        for id, label, helpText, handler in \
            [(wx.ID_NEW, '&New', 'New a file', self.OnNew),
             (wx.ID_OPEN, '&Open', 'Open a new file', self.OnOpen),
             (wx.ID_SAVE, '&Save', 'Save the current file', self.OnSave),
             (wx.ID_SAVEAS, 'Save &As', 'Save the file under a different name',self.OnSaveAs),
             (wx.ID_ABOUT, '&About', 'Information about this program',self.OnAbout),
             (None, None, None, None),
             (wx.ID_EXIT, 'E&xit', 'Terminate the program', self.OnExit)]:
            if id == None:
                fileMenu.AppendSeparator()
            else:
                item = fileMenu.Append(id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)
        menuBar.Append(fileMenu, '&File') # Add the fileMenu to the MenuBar
        self.SetMenuBar(menuBar)  # Add the menuBar to the Frame
        editMenu = wx.Menu()
        for id, label, helpText, handler in \
            [(wx.ID_CUT, '&cut', '', self.OnCut),
             (wx.ID_COPY, '&copy', '', self.OnCopy),
             (wx.ID_ANY, '&remove', 'remove file from list', self.OnRemove)]:
            item = editMenu.Append(id, label, helpText)
            self.Bind(wx.EVT_MENU, handler, item)
        menuBar.Append(editMenu,'&Edit')
        
    def CreateStatusAndTool(self):
        statusBar = self.CreateStatusBar() 
        toolBar = self.CreateToolBar()    
        self.toolQuit = toolBar.AddSimpleTool(wx.NewId(), wx.py.images.getPyBitmap(), "quit", "Quit,,,")
        toolBar.Realize()
        self.Bind(wx.EVT_TOOL, self.OnExit, self.toolQuit)
    
    def SetTitle(self):
        # MainWindow.SetTitle overrides wx.Frame.SetTitle, so we have to
        # call it using super:
        super(MainWindow, self).SetTitle('ECGLab: %s'%self.filename)
        
    def defaultFileDialogOptions(self):
        ''' Return a dictionary with file dialog options that can be
            used in both the save file dialog as well as in the open
            file dialog. '''
        return dict(message='Choose a file', defaultDir=self.dirname,
                    wildcard='*.*')
 
    def askUserForFilename(self, **dialogOptions):
        """Open and process a wxProject file."""
        dialog = wx.FileDialog(self, **dialogOptions)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True
            self.filename = dialog.GetFilename()
            self.dirname = dialog.GetDirectory()
            self.SetTitle() # Update the window title with the new filename
        else:
            userProvidedFilename = False
        dialog.Destroy()
        return userProvidedFilename
    
    # Event handlers:
    def OnNew(self, event):
        textfile = open(os.path.join(self.dirname, self.filename), 'w')
        textfile.write(self.control.GetValue())
        textfile.close()
        
    def OnExit(self, event):
        self.Close()  # Close the main window.
 
    def OnSave(self, event):
        textfile = open(os.path.join(self.dirname, self.filename), 'w')
        textfile.write(self.control.GetValue())
        textfile.close()
 
    def OnOpen(self, event):
        if self.askUserForFilename(style=wx.OPEN, **self.defaultFileDialogOptions()):
            textfile = open(os.path.join(self.dirname, self.filename), 'r')
            self.control.SetValue(textfile.read())
            textfile.close()
 
    def OnSaveAs(self, event):
        if self.askUserForFilename(defaultFile=self.filename, style=wx.SAVE, **self.defaultFileDialogOptions()):
            self.OnSave(event)
            
    def OnAbout(self, event):
        dialog = wx.MessageDialog(self, 'A sample editor\n' 'in wxPython', 'About Sample Editor', wx.OK)
        dialog.ShowModal()
        dialog.Destroy()
        
    def OnCut(self, event):
        pass
    
    def OnCopy(self, event):
        pass
    
    def OnRemove(self, event):
        pass  

if __name__ == '__main__':
    app = wx.App()
    frame = frame = MainWindow()
    frame.Center()
    frame.Show()
    app.MainLoop()