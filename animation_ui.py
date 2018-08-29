import sys
from PyQt5 import QtCore,uic
from PyQt5.QtWidgets import QMainWindow,QAction,QApplication,QFileDialog,qApp,QMessageBox
from PyQt5.QtGui import QIcon,QPixmap,QImage
import warnings
import numpy as np
import cv2
from past.builtins import xrange
import xlrd
import imutils
from PIL import Image
import os
import csv


# ignore warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------------
qtCreatorFile="animation.ui"

Ui_MainWindow,QtBaseClass=uic.loadUiType(qtCreatorFile)

class AnimationApp(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__() # super() returns the parent object of AnimationApp
        # QMainWindow.__init__(self)
        # Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.initUI()


    def initUI(self):
        self.reduceBtn.clicked.connect(self.showFinalPalette)
        self.reduceBtn2.clicked.connect(self.showFinalPalette_2)

        openAct=QAction(QIcon('open.png'),'&Open Image',self)
        openAct.setShortcut('Ctrl+O')
        openAct.triggered.connect(self.showImage)


        saveAct=QAction(QIcon('save.png'),'&Save',self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.triggered.connect(self.saveImg)

        exitAct=QAction(QIcon('exit.png'),'&Exit',self)
        exitAct.setShortcut('Ctrl+E')
        exitAct.triggered.connect(qApp.quit)

        menubar=self.menuBar()
        fileMenu=menubar.addMenu('&File')
        fileMenu.addAction(openAct)
        fileMenu.addAction(saveAct)
        fileMenu.addAction(exitAct)

        # toolbar
        openImageToolbar=self.addToolBar('&Open Image')
        saveToolbar = self.addToolBar('&Save')
        exitToolbar = self.addToolBar('&Exit')

        openImageToolbar.addAction(openAct)
        saveToolbar.addAction(saveAct)
        exitToolbar.addAction(exitAct)


        self.setWindowTitle('Color Palette Reduction')
        self.show()

    def showImage(self):

        self.fname=QFileDialog.getOpenFileName(self,'Open image','C:/Users/pandalgx/Dropbox/Animation analysis/animation image process/'
                                                                 'ui_design/animation3')
        if self.fname[0]:
            img = cv2.imread(self.fname[0])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # pixmap=QPixmap(self.fname[0])

            # # convert numpy array to QPixmap
            pixmap = self.np2qpixmap(img)
            w=self.originLabel.width()
            h=self.originLabel.height()

            self.originLabel.setPixmap(pixmap.scaled(w,h,aspectRatioMode=True))

            self.showAdobePalette()
        else:
            pass



    def showAdobePalette(self):
        fname = self.fname[0][:-4]+'.xlsx'

        if os.path.isfile(fname):
            self.adobePalette = self.extractPalette(fname)
            pixmap=self.np2qpixmap(self.adobePalette)

            w = self.paletteLabel1.width()
            h = self.paletteLabel1.height()
            self.paletteLabel1.setPixmap(pixmap.scaled(w, h, aspectRatioMode=True))

            self.colorQuantize()
        else:
            self.warningDialogue('There is no corresponding Excel File !')

    def warningDialogue(self,text):
        msg=QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle('Warning')
        msg.setText(text)

        msg.exec()


    def colorQuantize(self):
        clusters=self.adobePalette.reshape(-1,3)

        img = cv2.imread(self.fname[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # color quantization
        img_quan = self.image_quantize(img, clusters)

        # # convert numpy array to QPixmap
        pixmap=self.np2qpixmap(img_quan)

        w = self.quantizeLabel.width()
        h = self.quantizeLabel.height()
        self.quantizeLabel.setPixmap(pixmap.scaled(w,h,aspectRatioMode=True))

        self.psnr(img_quan,self.psnrLabel1)

        pass


    # white excluded
    def showFinalPalette(self):
        if self.fname[0]:
            clusters = self.adobePalette.reshape(-1, 3)
            clusters_Luv = cv2.cvtColor(clusters.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2Luv).reshape(-1, 3)
            clusters_Luv = clusters_Luv.astype(np.int32)
            distance=int(self.vector_distance.text())
            self.clusters_final=self.final_reduced(clusters,clusters_Luv,distance)
            # get palette
            k=self.clusters_final.shape[0]
            self.numLabel.setText('('+str(k)+')')
            quotient = int(k / 5)
            # display palette with different rows
            if k%5==0:
                palette = self.clusters_final.reshape(-1, 5, 3)
            elif k%5==4:
                palette_1=self.clusters_final[0:5*quotient].reshape(-1,5,3)
                palette_2=self.clusters_final[5*quotient:k].reshape(1,-1,3)
                white = (255 * np.ones((1, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette=np.vstack((palette_1,palette_2))
            elif k%5==3:
                palette_1=self.clusters_final[0:5*quotient].reshape(-1,5,3)
                palette_2=self.clusters_final[5*quotient:k].reshape(1,-1,3)
                white=(255*np.ones((2,3)).astype(int)).reshape(1,-1,3)
                palette_2=np.hstack((palette_2,white))
                palette=np.vstack((palette_1,palette_2))
            elif k%5==2:
                palette_1=self.clusters_final[0:5*quotient].reshape(-1,5,3)
                palette_2=self.clusters_final[5*quotient:k].reshape(1,-1,3)
                white = (255 * np.ones((3, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette=np.vstack((palette_1,palette_2))
            elif k%5==1:
                palette_1=self.clusters_final[0:5*quotient].reshape(-1,5,3)
                palette_2=self.clusters_final[5*quotient:k].reshape(1,-1,3)
                white = (255 * np.ones((4, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette=np.vstack((palette_1,palette_2))
            pixmap = self.np2qpixmap(palette)
            self.finalPalette=palette

            w = self.paletteLabel2.width()
            h = self.paletteLabel2.height()
            self.paletteLabel2.setPixmap(pixmap.scaled(w, h, aspectRatioMode=True))

            self.colorQuantize_2()

            self.showReducedValue(self.clusters_final)
        else:
            self.warningDialogue('Image is not selected!\nPlease select an image!')


    # white included
    def showFinalPalette_2(self):
        if self.fname[0]:
            clusters = self.adobePalette.reshape(-1, 3)
            clusters_Luv = cv2.cvtColor(clusters.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2Luv).reshape(-1, 3)
            clusters_Luv = clusters_Luv.astype(np.int32)
            distance = int(self.vector_distance.text())
            self.clusters_final = self.final_reduced_2(clusters, clusters_Luv, distance)
            # get palette
            k = self.clusters_final.shape[0]
            self.numLabel.setText('(' + str(k) + ')')
            quotient = int(k / 5)
            # display palette with different rows
            if k % 5 == 0:
                palette = self.clusters_final.reshape(-1, 5, 3)
            elif k % 5 == 4:
                palette_1 = self.clusters_final[0:5 * quotient].reshape(-1, 5, 3)
                palette_2 = self.clusters_final[5 * quotient:k].reshape(1, -1, 3)
                white = (255 * np.ones((1, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette = np.vstack((palette_1, palette_2))
            elif k % 5 == 3:
                palette_1 = self.clusters_final[0:5 * quotient].reshape(-1, 5, 3)
                palette_2 = self.clusters_final[5 * quotient:k].reshape(1, -1, 3)
                white = (255 * np.ones((2, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette = np.vstack((palette_1, palette_2))
            elif k % 5 == 2:
                palette_1 = self.clusters_final[0:5 * quotient].reshape(-1, 5, 3)
                palette_2 = self.clusters_final[5 * quotient:k].reshape(1, -1, 3)
                white = (255 * np.ones((3, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette = np.vstack((palette_1, palette_2))
            elif k % 5 == 1:
                palette_1 = self.clusters_final[0:5 * quotient].reshape(-1, 5, 3)
                palette_2 = self.clusters_final[5 * quotient:k].reshape(1, -1, 3)
                white = (255 * np.ones((4, 3)).astype(int)).reshape(1, -1, 3)
                palette_2 = np.hstack((palette_2, white))
                palette = np.vstack((palette_1, palette_2))
            pixmap = self.np2qpixmap(palette)
            self.finalPalette = palette

            w = self.paletteLabel2.width()
            h = self.paletteLabel2.height()
            self.paletteLabel2.setPixmap(pixmap.scaled(w, h, aspectRatioMode=True))

            self.colorQuantize_2()

            self.showReducedValue(self.clusters_final)
        else:
            self.warningDialogue('Image is not selected!\nPlease select an image!')


    def colorQuantize_2(self):

        img = cv2.imread(self.fname[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # color quantization
        img_quan = self.image_quantize(img, self.clusters_final)
        self.img_quan_final=img_quan

        # # convert numpy array to QPixmap
        pixmap = self.np2qpixmap(img_quan)

        w = self.quantizeLabel2.width()
        h = self.quantizeLabel2.height()
        self.quantizeLabel2.setPixmap(pixmap.scaled(w, h, aspectRatioMode=True))

        self.psnr(img_quan,self.psnrLabel2)

        pass

    def psnr(self,img_quan,psnrLabel):
        orig=cv2.imread(self.fname[0])
        orig=cv2.cvtColor(orig,cv2.COLOR_RGB2Luv).astype(np.float32)
        M,N,_=orig.shape
        img_quan=img_quan.astype(np.uint8)
        img_quan=cv2.cvtColor(img_quan,cv2.COLOR_RGB2Luv).astype(np.float32)

        mseL=np.mean((orig[:,:,0]-img_quan[:,:,0])**2)
        mseU = np.mean((orig[:, :, 1] - img_quan[:, :, 1]) ** 2)
        mseV = np.mean((orig[:, :, 2] - img_quan[:, :, 2]) ** 2)
        mse=(mseL+mseU+mseV)/3

        PSNR=10*np.log10(255**2/mse)
        psnrLabel.setText('(PSNR=%.2fdB)'% PSNR)



    def saveImg(self):
        name=QFileDialog.getSaveFileName(self,'Save Image','data/save','Images (*.jpg *.png)')
        self.csvName=name[0][:-3]+'csv'

        if name[0]:
            result2=Image.new("RGB",(400,400),color=(255,255,255))
            img2=self.img_quan_final.astype(np.uint8)
            img2=Image.fromarray(img2)
            img2.thumbnail((400,400),Image.ANTIALIAS)
            w1,h1=img2.size
            result2.paste(img2,(0,0,w1,h1))

            final_palette=self.finalPalette.astype(np.uint8)
            final_palette=imutils.resize(final_palette,width=160)
            final_palette=Image.fromarray(final_palette)
            w2,h2=final_palette.size
            result2.paste(final_palette,(w1//4,h1+40,w1//4+w2,h1+40+h2))
            # result2.show()
            result2.save(name[0])

            self.saveCsv()
        else:

            pass


    def saveCsv(self):

        clusters_final_value=np.vstack((np.array([' R',' G',' B']),self.clusters_final.astype(int)))

        with open(self.csvName,'w',newline='') as csvfile:
            csvWriter=csv.writer(csvfile)
            csvWriter.writerows(clusters_final_value)
        pass


# ----------------------------
# following are packages
# ----------------------------
    # function to quantize image color in Luv domain
    def image_quantize(self, img, clusters):
        # convert clusters from rgb to Luv
        clusters_Luv = cv2.cvtColor(clusters.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2Luv).reshape(-1, 3)
        clusters_Luv = clusters_Luv.astype(np.int32)

        # convert image from rgb to Luv
        img_Luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
        N, M, _ = img.shape

        # reshape img from N*M*3 to NM*3
        img_Luv_vector = img_Luv.reshape(N * M, -1)
        # calculate errors between img_vector and each vector in clusters
        errors = np.zeros((N * M, 3, clusters.shape[0]))
        for i in xrange(clusters.shape[0]):
            errors[:, :, i] = img_Luv_vector - clusters_Luv[i]

        errors_norm = np.linalg.norm(errors, axis=1, keepdims=True)
        # classify the image vectors into nearest cluster_vector
        vector_index = np.argmin(errors_norm, axis=2).flatten()
        # image quantization using vector_index
        img_quan_vector = clusters[vector_index]
        # reshape quantized image vector to image shape
        img_quan = img_quan_vector.reshape((N, M, -1))

        return img_quan


    # convert numpy array to QPixmap
    def np2qpixmap(self, img):
        img = img.astype(np.uint8)
        h, w, c = img.shape

        pixmap = QImage(img, w, h,3*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)

        return pixmap

    def extractPalette(self,excelPath):
        xls = xlrd.open_workbook(excelPath)

        table = xls.sheet_by_index(0)
        clusters = np.zeros([25, 3])

        for i in range(25):
            vector = table.row_values(i)
            clusters[i] = vector

        palette = clusters.reshape(5, -1, 3)

        return palette

    # reduce cluster vectors based on their distance (white excluded)
    def reduce_color(self, clusters, clusters_Luv, index, vector_distance):
        errors_2 = clusters_Luv - clusters_Luv[index]
        errors_2_norm = np.linalg.norm(errors_2, axis=1)
        # reduce white vectors
        indexs1 = np.where(np.all(clusters >= 239, axis=1))
        clusters = np.delete(clusters, indexs1[0], 0)
        clusters_Luv = np.delete(clusters_Luv, indexs1[0], 0)
        # reduce similar vectors
        indexs2 = np.where(errors_2_norm <= vector_distance)
        indexs3=list(indexs2[0])
        indexs3.remove(index)
        clusters_reduced = np.delete(clusters, indexs3,axis=0)  # remove similar vectors
        clusters_Luv_reduced = np.delete(clusters_Luv, indexs3, axis=0)

        return clusters_reduced, clusters_Luv_reduced


    # calculate final clusters (white excluded)
    def final_reduced(self, clusters, clusters_Luv, vector_distance):
        loop_num = clusters.shape[0]
        index = 0
        while index < loop_num:
            clusters, clusters_Luv = self.reduce_color(clusters, clusters_Luv, index, vector_distance)
            index += 1
            loop_num = clusters.shape[0]

        return clusters

    # reduce cluster vectors based on their distance (white included)
    def reduce_color_2(self, clusters, clusters_Luv, index, vector_distance):
        errors_2 = clusters_Luv - clusters_Luv[index]
        errors_2_norm = np.linalg.norm(errors_2, axis=1)
        # # reduce white vectors
        # indexs1 = np.where(np.all(clusters >= 239, axis=1))
        # clusters = np.delete(clusters, indexs1[0], 0)
        # clusters_Luv = np.delete(clusters_Luv, indexs1[0], 0)
        # reduce similar vectors
        indexs2 = np.where(errors_2_norm <= vector_distance)
        indexs3 = list(indexs2[0])
        indexs3.remove(index)
        clusters_reduced = np.delete(clusters, indexs3, axis=0)  # remove similar vectors
        clusters_Luv_reduced = np.delete(clusters_Luv, indexs3, axis=0)

        return clusters_reduced, clusters_Luv_reduced

    # calculate final clusters (white included)
    def final_reduced_2(self, clusters, clusters_Luv, vector_distance):
        loop_num = clusters.shape[0]
        index = 0
        while index < loop_num:
            clusters, clusters_Luv = self.reduce_color_2(clusters, clusters_Luv, index, vector_distance)
            index += 1
            loop_num = clusters.shape[0]

        return clusters


    # show reduced color rgb value
    def showReducedValue(self,clusters_final):
        # initialization
        for i in xrange(25):
            r=getattr(self,'r{0}'.format(i+1))
            r.setText('')

            g = getattr(self, 'g{0}'.format(i + 1))
            g.setText('')

            b = getattr(self, 'b{0}'.format(i + 1))
            b.setText('')

        num_colors=clusters_final.shape[0]
        for i in xrange(num_colors):
            r=getattr(self,'r{0}'.format(i+1))
            r.setText('%d'%clusters_final[i,0])

            g = getattr(self, 'g{0}'.format(i + 1))
            g.setText('%d' % clusters_final[i, 1])

            b = getattr(self, 'b{0}'.format(i + 1))
            b.setText('%d' % clusters_final[i, 2])


# Main
app=QApplication(sys.argv) # Every PyQt5 application must create an application object
aniApp=AnimationApp()
sys.exit(app.exec_()) # exec_() - enters mainloop of the application



