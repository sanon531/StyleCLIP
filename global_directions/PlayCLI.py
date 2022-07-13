import clip
import matplotlib
import cv2
import dlib
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms

from tkinter import Tk 
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from GUI import View
from Inference import StyleCLIP
from manipulate import Manipulator
from MapTS import GetFs,GetBoundary,GetDt
from argparse import Namespace
from encoder4editing.models.psp import pSp
from utils.alignment import align_face



#%%


class PlayCLI():  #Controller
    '''
    followed Model View Controller Design Pattern
    
    controller, model, view
    '''
    def __init__(self,dataset_name='ffhq',origin_pic_address ="data/ffhq/0.jpg",alpha = 8,beta = 0.15, skip = True ):
        matplotlib.use('Agg')
        #prework
        self.dataset_name = dataset_name
        self.origin_pic_address = origin_pic_address
        
        #베타값과 알파값 설정
        self.img_ratio=2

        print("Successfully Image Inputted: alpha : ",alpha ,"beta :",beta)
        self.manipulationStrength = alpha
        self.disentangleThreshold = beta
        self.origin_pic = Image.open(origin_pic_address)
        self.origin_pic = self.origin_pic.convert("RGB")


        if not skip :
            #self.AlignImage(origin_pic_address)
            self.BuildLatent()
            self.style_clip=StyleCLIP(dataset_name)
            self.GetGUIData()
        else:
            self.style_clip=StyleCLIP(dataset_name)

            

        self.InputTextDescription()
        self.SetInit()
        #이부분을 루프 시켜서 중간에 계속해서 대입을 시켜 볼 수 있도록 만들것이다.
        self.LoopListOfPlay()
        #self.open_img()

        # 지금 해야할꺼는 내가 바꾸고 싶은 이미지를 기반으로 한번 만들어보는것이다.
        # 지금은 약간 손으로 일일히 이동 시켜줘야한다 레이턴트의 형성도 자동으로 되지않고 
        # 그래서 레이턴트를 여기서 만들고할수있도록 pt를 다운 받아서 하게 되었고 
        # 7월 13일날 이제 해당하는 부분을 끼워 넣은 뒤 해보도록 하자

        # 루프 돌릴때 원래 이미지로 초기화 안되는 이슈 수정하기.
        while True:
            keepChange = 'N'
            print("Need Change alpha and beta on this image? [y/N]:")
            keepChange = input()
            if str(keepChange) =='N' or  str(keepChange) =='n':
                exit()
            elif str(keepChange) =='Y' or  str(keepChange) =='y':
                #여기 안에 넣을만한 변수들로 수정하기.
                print("Alpha:")
                self.manipulationStrength = float(input())
                print("Beta:")
                self.disentangleThreshold = float(input())
                self.LoopListOfPlay()
            else : 
                print("Error!: Retype query please!")

 
        self.drawn  = None

        #self.origin_pic.save(result_pic)

        exit()
        
        

    def LoopListOfPlay(self):

        self.ChangeAlpha()
        self.ChangeBeta()
        img=self.style_clip.GetImg()
        img = Image.fromarray(img)
        img.save("alala.jpg")

    def AlignImage(self, origin_pic_address):
        print("Align Image")
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.origin_pic = align_face(filepath=origin_pic_address, predictor=predictor) 
  

    def BuildLatent(self) :
        print("Build latent : start")
        EXPERIMENT_ARGS = {
            "model_path": "e4e_ffhq_encode.pt"
        }
        model_path = EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts= Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()
        self.net.cuda()
        print("Build latent : net Set")

        img_transforms = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 256)       
        transformed_image = img_transforms(self.origin_pic)
        print("Build latent : Image Transformed")

        inputs = transformed_image.unsqueeze(0)
        experiment_type = 'ffhq_encode'
        #위에 까지가 Align하는부분 아래부분이 이제 진짜 인버트 하는 부분
        self.origin_pic, self.latents = self.net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        print("Build latent : latent processed")

        torch.save(self.latents, 'data/ffhq/latents.pt')
        self.w_plus=self.latents.cpu().detach().numpy()
        np.save('./data/ffhq/w_plus.npy',self.w_plus)
        #pt 가 생성되었으니 GUI data 그래대로 연결하는게 좋을듯.

    # latent, inverted image 만드는 곳.
    def GetGUIData(self):
        # real 기반으로 일단 만들어봄
        tmp=self.style_clip.M.W2S(self.w_plus)
        self.style_clip.M.dlatents=tmp

        self.style_clip.M.img_index=0
        self.style_clip.M.num_images=len(self.w_plus)
        self.style_clip.M.alpha=[0]
        self.style_clip.M.step=1
        lindex,bname=0,0
    

        

        # 그리고 이뒤에 GetGUI에서 정리해주는 코드 뺴와서 하자




    # 이부분도 인풋으로 작업할 수 있을지 한번 봐보고 싶었는데 일단은
    def InputTextDescription(self):
        print('Set TextDescription')
        neutral='face' 
        target=' Smilling face' 
        self.style_clip.target=neutral
        self.style_clip.target=target
        self.style_clip.GetDt2()

    def SetInit(self):
        codes=self.style_clip.GetCode()
        self.style_clip.M.dlatent_tmp=[tmp[:,0] for tmp in codes]
        print('set init')
    
    def ChangeAlpha(self):
        self.style_clip.M.alpha=[self.manipulationStrength]
        
        img=self.style_clip.GetImg()
        print('manipulate Alpha : ', self.manipulationStrength)
        img=Image.fromarray(img)
        img.save("result_AFTERaLPHA1.jpg")

        self.addImage_m(img)
        
    def ChangeBeta(self):
        self.style_clip.beta=float(self.disentangleThreshold)
        
        img=self.style_clip.GetImg()
        print('manipulate Beta:',self.disentangleThreshold)
        img=Image.fromarray(img)
        img.save("result_AFTERBETA1.jpg")
        
        self.addImage_m(img)

    def ChangeDataset(self,event):
        dataset_name=self.dataset_name
        self.style_clip.LoadData(dataset_name)
    
    def Reset(self):
        self.style_clip.GetDt2()
        

    
    def addImage(self,img):
        #self.view.bg.create_image(self.view.width/2, self.view.height/2, image=img, anchor='center')
        self.image=img #save a copy of image. if not the image will disappear
        
    def addImage_m(self,img):
        #self.view.mani.create_image(512, 512, image=img, anchor='center')
        self.image2=img
        
    
    def openfn(self):
        filename = askopenfilename(title='open',initialdir='./data/'+self.style_clip.M.dataset_name+'/',filetypes=[("all image format", ".jpg"),("all image format", ".png")])
        return filename
    
    def open_img(self):
        x = self.openfn()
        print(x)
        
        
        img = Image.open(x)
        img2 = img.resize(( 512,512), Image.ANTIALIAS)
        #img2 = ImageTk.PhotoImage(img2)
        self.addImage(img2)
        
        #img = ImageTk.PhotoImage(img)
        self.addImage_m(img)
        
        img_index=x.split('/')[-1].split('.')[0]
        img_index=int(img_index)
        print(img_index)
        self.style_clip.M.img_index=img_index
        self.style_clip.M.dlatent_tmp=[tmp[img_index:(img_index+1)] for tmp in self.style_clip.M.dlatents]
        
        self.style_clip.GetDt2()
        
    #%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_name',type=str,default='ffhq',help='name of dataset, for example, ffhq')
    parser.add_argument('--origin_pic_address',type=str,default='data/ffhq/0.jpg',help='Address of original image')
    parser.add_argument('--alpha',type=float,default=7.5,help='manipulation strength')
    parser.add_argument('--beta',type=float,default=0.15,help='disentangle threshold')
    parser.add_argument('--skip',type=str,default='True',help='SkipLatent Or Other')
   
    args = parser.parse_args()
    dataset_name=args.dataset_name
    origin_pic_address = args.origin_pic_address 
    alpha = args.alpha
    beta =args.beta

    skip = True
    if(args.skip == 'False'):
        skip = False


    self=PlayCLI(dataset_name,origin_pic_address,alpha,beta,skip)
    self.run()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    