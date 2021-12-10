#imports required for flask part
from flask import Flask, render_template,url_for, request
from werkzeug.utils import secure_filename
import os

#BERT imports required for AI part
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np

#OCR part imports for memes
import pytesseract
try:
 from PIL import Image
except ImportError:
 import Image

#video part imports
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi


app = Flask(__name__)

@app.route("/")
def msg():
    return render_template('index.html')

@app.route("/tt.html",methods = ['POST', 'GET'])
def text():
    if(request.method=='GET'):
        return render_template('tt.html')
    else:
        f = request.files['file']
        f.save(secure_filename(f.filename))
        
        #model path-BERT-1
        model_path1="bert-1-model-1-e"

        #model path-BERT2
        model_path2="bert-2-model-2"

        #test file path
        file_path=f.filename

        #tokenizer 
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')       

        #read the csv file uploaded
        df=pd.read_csv(file_path,encoding='latin1')

        #column names shuld only be Text and Category in the csv file uploaded
        x_test=df['Text']
        

        #predict for all the texts in the csv file using the saved model and save the predictions in y_pred list
        
        #predict using the loaded model
        loaded_model1 = TFDistilBertForSequenceClassification.from_pretrained(model_path1)
        y_pred=[]

        #3-normal and 5-sarcasm
        non_hate=[3,5]

        for test_sentence in x_test:
            predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")
            tf_output = loaded_model1.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()
            ans=np.argmax(tf_prediction,axis=1)
            if ans in non_hate:
                ans='N' #N-No-it is not hate content 
            else:
                ans='Y' #Y-yes-it is hate content
            y_pred.append(ans)

        df['Hate Classification']=y_pred

        #give name of req csv file
        df.to_csv('predicted1.csv')

        #goto BERT2 if hate content is identified        
        df2=pd.read_csv('predicted1.csv')
        df2[df2['Hate Classification']=='Y']['Text']

        #keep only hate content
        df2=df2[df2['Hate Classification']=='Y']
        x_test2=df2[df2['Hate Classification']=='Y']['Text']

        #predict using the loaded model-BERT-2
        loaded_model2 = TFDistilBertForSequenceClassification.from_pretrained(model_path2)

        y_pred2=[]

        for test_sentence in x_test2:
            predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")
            tf_output = loaded_model2.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()
            ans=np.argmax(tf_prediction,axis=1)
            cat=ans[0]
            if(cat==0):
                res="toxic"
            elif(cat==1):
                res="severely toxic"
            elif(cat==2):
                res="obscene"
            elif(cat==3):
                res="threat"
            elif(cat==4):
                res="insult"
            else:
                res="identity_hate"
            y_pred2.append(res)

        #add Toxicity column
        df2['Toxicity']=y_pred2

        #file name-RESULTS are stored here
        filename = 'predicted2.csv' 

        #give name of req csv file-predicted2.csv is the final csv file with hate content identified and toxicity level of the corr hate content
        df2.to_csv(filename)    
 
	# to read the csv file using the pandas library 
        data = pd.read_csv(filename,header=0) 
 
        myData = data.values 
        return render_template('bullies.html', myData=myData) 
        

@app.route("/img.html",methods=['POST','GET'])
def img():
    if(request.method=='GET'):
        return render_template('img.html')
    else:
        f = request.files['file']
        f.save(secure_filename(f.filename))
        
        #model path
        model_path1="bert-1-model-1-e"

        #model path-BERT2
        model_path2="bert-2-model-2"

        #path of the meme/image uploaded by the user
        image_path=f.filename

        #tokenizer 
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        def predict(model_path,extractedInformation):
            test_sentence = extractedInformation
            loaded_model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
            predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")
            tf_output = loaded_model.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()
            ans=np.argmax(tf_prediction,axis=1)
            return ans[0]

        #set path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        #do OCR using the pytesseract library
        extractedInformation = pytesseract.image_to_string(Image.open(image_path))       

        non_hate_codes=[3,5]

        ans1=predict(model_path1,extractedInformation)

        if ans1 in non_hate_codes:
            msg="Not Hate"
            
        else:
            msg1="Hate||"
            msg2="Toxicity level: "
            cat=predict(model_path2,extractedInformation)
            if(cat==0):
                res="toxic"
            elif(cat==1):
                res="severely toxic"
            elif(cat==2):
                res="obscene"
            elif(cat==3):
                res="threat"
            elif(cat==4):
                res="insult"
            else:
                res="identity_hate"
            msg=msg1+msg2+res
                
        return msg

@app.route("/vidcheck.html",methods = ['POST', 'GET'])
def video():
    if(request.method=='GET'):
        return render_template('vidcheck.html')
    else:
        playlist_url = request.form['purl']        
        
        #model path-BERT-1
        model_path1="bert-1-model-1-e"

        #model path-BERT2
        model_path2="bert-2-model-2"

        api_key='AIzaSyBxBGwzMStykP4NM7gmrJQ5vjwKNDbdeSs'
        youtube=build('youtube','v3',developerKey=api_key)

        def get_video_ids(youtube, playlist_id):
            request = youtube.playlistItems().list(part='contentDetails',playlistId = playlist_id,maxResults = 50)
            response = request.execute()
    
            video_ids = []
    
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])
                next_page_token = response.get('nextPageToken')
                more_pages = True
            while more_pages:
                if next_page_token is None:
                    more_pages = False
                else:
                    request = youtube.playlistItems().list(part='contentDetails',playlistId = playlist_id,maxResults = 50,pageToken = next_page_token)
                    response = request.execute()
                    for i in range(len(response['items'])):
                        video_ids.append(response['items'][i]['contentDetails']['videoId'])            
                    next_page_token = response.get('nextPageToken')
        
            return video_ids

        pos=playlist_url.find('=')+1
        playlist_id=playlist_url[pos:]
        video_ids=get_video_ids(youtube, playlist_id)

        def generate_transcript(id):
            transcript = YouTubeTranscriptApi.get_transcript(id)
            script = ""
            for text in transcript:
                t=text["text"]
                if t!='[Music]':
                    script+= t+" "
            return script,len(script.split())
        video_links=[]
        transcripts=[]
        for vid in video_ids:
            video_link="https://www.youtube.com/watch?v="+vid
            try:
                transcript, no_of_words = generate_transcript(vid)
                transcripts.append(transcript)
                video_links.append(video_link)
            except Exception:
                continue

        df=pd.DataFrame(list(zip(video_links,transcripts)),columns=['Video URL','Transcript'])
        df.to_csv('videos.csv')


        #test file path
        file_path='videos.csv'

        #tokenizer 
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')       

        #read the csv file uploaded
        df=pd.read_csv(file_path,encoding='latin1')

        #column names shuld only be Text and Category in the csv file uploaded
        x_test=df['Transcript']
        

        #predict for all the texts in the csv file using the saved model and save the predictions in y_pred list
        
        #predict using the loaded model
        loaded_model1 = TFDistilBertForSequenceClassification.from_pretrained(model_path1)
        y_pred=[]

        #3-normal and 5-sarcasm
        non_hate=[3,5]

        for test_sentence in x_test:
            predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")
            tf_output = loaded_model1.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()
            ans=np.argmax(tf_prediction,axis=1)
            if ans in non_hate:
                ans='N' #N-No-it is not hate content 
            else:
                ans='Y' #Y-yes-it is hate content
            y_pred.append(ans)

        df['Hate Classification']=y_pred

        #give name of req csv file
        df.to_csv('predicted1.csv')

        #goto BERT2 if hate content is identified        
        df2=pd.read_csv('predicted1.csv')        

        #keep only hate content
        df2=df2[df2['Hate Classification']=='Y']
        x_test2=df2['Transcript']

        #predict using the loaded model-BERT-2
        loaded_model2 = TFDistilBertForSequenceClassification.from_pretrained(model_path2)

        y_pred2=[]

        for test_sentence in x_test2:
            predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")
            tf_output = loaded_model2.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()
            ans=np.argmax(tf_prediction,axis=1)
            cat=ans[0]
            if(cat==0):
                res="toxic"
            elif(cat==1):
                res="severely toxic"
            elif(cat==2):
                res="obscene"
            elif(cat==3):
                res="threat"
            elif(cat==4):
                res="insult"
            else:
                res="identity_hate"
            y_pred2.append(res)

        #add Toxicity column
        df2['Toxicity']=y_pred2

        #file name-RESULTS are stored here
        filename = 'predicted2.csv' 

        #give name of req csv file-predicted2.csv is the final csv file with hate content identified and toxicity level of the corr hate content
        df2.to_csv(filename)    
 
	# to read the csv file using the pandas library 
        data = pd.read_csv(filename,header=0) 
 
        myData = data.values 
        return render_template('videos.html', myData=myData) 


        

@app.route("/home.html")
def home():
    return render_template('home.html')

@app.route("/about.html")
def about():
    return render_template('about.html')
    


if __name__ =="__main__":
    app.run(debug=True,port=8000)
