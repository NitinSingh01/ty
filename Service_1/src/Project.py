#!/usr/bin/python
# coding: utf-8



# Import libraries
from pydub import AudioSegment
import io
import os
from google.cloud import speech_v1p1beta1 as speech
import wave
import urllib.request
from google.cloud import storage
from werkzeug.utils import secure_filename
from flask import jsonify, request #Flask for Restful API
import config #API Config
from flask import Flask
from rq import Queue, Worker, Connection
from rq.job import Job
from google.cloud import language_v1

# Google Secret Manager
import json
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
secret_name = "Voice"
project_id = "speech-1609595980955"
req = {"name": f"projects/{project_id}/secrets/{secret_name}/versions/latest"}
response = client.access_secret_version(req)
secret = response.payload.data.decode("UTF-8")
s = json.loads(secret)


filepath = s.get("filepath")      #Input audio file path
filepath1 = s.get("filepath1")          
output_filepath = s.get("output_filepath") #Final transcript path
bucketname = s.get("bucketname") #Name of the bucket created in the step before


app = Flask(__name__) #creates the Flask instance


# Function to pass Audio file for processing
def translate_main(audio_file_name):
    transcript = google_transcribe(audio_file_name)
    transcript_filename = audio_file_name.split('.')[0] + '_transcript' + '.txt'
    write_transcripts(transcript_filename,transcript)
    word_details = google_word_details(audio_file_name)
    word_details_filename = transcript_filename.split('.')[0] + '_word_details' + '.txt'
    write_word_details(word_details_filename,word_details)
    sentiment = analyze_sentiment(transcript_filename)
    sentiment_filename = transcript_filename.split('.')[0] + '_sentiment' + '.txt'
    write_sentiment(sentiment_filename,sentiment)

def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")

def frame_rate_channel(audio_file_name):
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        return frame_rate,channels

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

# pass filename
ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API to pass audio file
@app.route('/app/v1/file-upload', methods=['GET','POST'])
def upload_file():
    if request.method == "POST":
        from Project import translate_main
    # check if the post request has the file part
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message' : 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('/home/asheeshg01/input', filename))
            redis_url = os.getenv('REDISTOGO_URL','redis://redis:8080')
            conn = redis.from_url(redis_url)
            q = Queue(conn)
            job = q.enqueue_call(func=translate_main, args=(file.filename,), result_ttl=50000)
            print(job.get_id())
            resp = jsonify({'Job Id to access details is' : job.get_id()})
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({'message' : 'Allowed file types is wav'})
            resp.status_code = 400
            return resp

# Function to convert speech-to-text
def google_transcribe(audio_file_name):
    file_name = filepath + audio_file_name
    second_lang = "hi-IN"
   

    # The name of the audio file to transcribe
    
    frame_rate, channels = frame_rate_channel(file_name)
    
    if channels > 1:
        stereo_to_mono(file_name)
    
    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name
    
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''
        
    credential_path = s.get("credential_path")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=frame_rate,
    language_code='en-US',
    alternative_language_codes=[second_lang],
    enable_speaker_diarization=True,
    diarization_speaker_count=2)

  
    operation = client.long_running_recognize(request={"config":config, "audio":audio})
    response = operation.result(timeout=10000)
    result = response.results[-1]
    words_info = result.alternatives[0].words
    
    tag=1
    speaker=""

    for word_info in words_info:
        if word_info.speaker_tag==tag:
            speaker=speaker+" "+word_info.word
        else:
            transcript += "speaker {}: {}".format(tag,speaker) + '\n'
            tag=word_info.speaker_tag
            speaker=""+word_info.word
          
    
    transcript += "speaker {}: {}".format(tag,speaker)
    #for result in response.results:
        #transcript += result.alternatives[0].transcript
    
    storage_client = storage.Client()
    bucket_name = storage_client.get_bucket(bucket_name)
    transcript_filename = audio_file_name.split('.')[0] + '_transcript' + '.txt'
    blob_transcript_file = bucket_name.blob(transcript_filename) 
    blob_transcript_file.upload_from_string(transcript)

    #delete_blob(bucket_name, destination_blob_name)
    return transcript


# Function to retrieve word details

def google_word_details(audio_file_name):
    file_name = filepath + audio_file_name
    second_lang = "hi-IN"
    frame_rate, channels = frame_rate_channel(file_name)
    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''
    word_details = ''
    credential_path = s.get("credential_path")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=frame_rate,
    language_code='en-US',
    alternative_language_codes=[second_lang],
    enable_speaker_diarization=True,
    diarization_speaker_count=2,
    enable_word_time_offsets=True)

    # Detects speech in the audio file
    #operation = client.long_running_recognize(config, audio)
    
    operation = client.long_running_recognize(request={"config":config, "audio":audio})
    response = operation.result(timeout=10000)
    result = response.results[-1]
    words_info = result.alternatives[0].words
    
    tag=1
    speaker=""

    for word_info in words_info:
        word = word_info.word
        start_time = word_info.start_time
        end_time = word_info.end_time
        speaker1 = word_info.speaker_tag
        word_details += " Word: {} : start_time: {}: end_time: {}: speaker {}".format(word,start_time.total_seconds(),end_time.total_seconds(),speaker1)
    
    storage_client = storage.Client()
    bucket_name = storage_client.get_bucket(bucket_name)
    word_details_filename = audio_file_name.split('.')[0] + '_word_details' + '.txt'
    blob_word_details_file = bucket_name.blob(word_details_filename) 
    blob_word_details_file.upload_from_string(word_details)    
    
    #delete_blob(bucket_name, destination_blob_name)
    return word_details


# Write Speech-to-Text transcript to output file
def write_transcripts(transcript_filename,transcript):
    f= open(output_filepath + transcript_filename,"w+")
    f.write(transcript)
    f.close()


# Write Sentiment details to output file
def write_sentiment(sentiment_filename,sentiment):
    f= open(output_filepath + sentiment_filename,"w+")
    f.write(sentiment)
    f.close()


# Write Word details to output file
def write_word_details(word_details_filename,word_details):
    f= open(output_filepath + word_details_filename,"w+")
    f.write(word_details)
    f.close()


# Function to retrieve sentiment of the conversation
def analyze_sentiment(transcript_filename):
    """
    Analyzing Sentiment in text file stored in Cloud Storage

    Args:
      gcs_content_uri Google Cloud Storage URI where the file content is located.
      e.g. gs://[Your Bucket]/[Path to File]
    """
    credential_path = s.get("credential_path")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    client = language_v1.LanguageServiceClient()
    
    file_name = filepath1 + transcript_filename
    bucket_name = bucketname
    source_tr_file_name = filepath1 + transcript_filename
    destination_tr_blob_name = transcript_filename
    
    upload_blob(bucket_name, source_tr_file_name, destination_tr_blob_name)
    
    gcs_content_uri = 'gs://' + bucketname + '/' + transcript_filename
   
    

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"gcs_content_uri": gcs_content_uri, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Get overall sentiment of the input document
    
    sentiment_score= "Document sentiment score: {}".format(response.document_sentiment.score)
    sentiment_magnitude= "Document sentiment magnitude: {}".format(response.document_sentiment.magnitude)
    sentiment = "{} and {}".format(sentiment_score,sentiment_magnitude)

    storage_client = storage.Client()
    bucket_name = storage_client.get_bucket(bucket_name)
    sentiment_filename = transcript_filename.split('.')[0] + '_sentiment' + '.txt'
    blob_sentiment_file = bucket_name.blob(sentiment_filename) 
    blob_sentiment_file.upload_from_string(sentiment)

    #delete_blob(bucket_name, destination_tr_blob_name)
    return sentiment

# API to retreive output based on Job Id
@app.route("/app/v1/<job_key>", methods=['GET'])
def get_results(job_key):
    q= Queue(connection=conn)
    job = q.fetch_job(job_key)

    if job.is_finished:
       return "Yey,your job is processed.Please see the results in the directory!", 201        
    else:
        return "Still job is under progress.Please try after sometime", 202

# Main function

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)

#!/usr/bin/python
# coding: utf-8

# In[83]:


# Import libraries
from pydub import AudioSegment
import io
import os
import json
#from google.cloud import speech
#from google.cloud.speech import enums
#from google.cloud.speech import types
from google.cloud import speech_v1p1beta1 as speech #Changed
#from google.cloud.speech_v1p1beta1 import enums #Changed
#from google.cloud.speech_v1p1beta1 import types #Changed
import wave
import urllib.request
from google.cloud import storage
from werkzeug.utils import secure_filename
from flask import jsonify, request #Flask for Restful API
import config #API Config
from flask import Flask
from rq import Queue
from rq.job import Job
from 	
from google.cloud import language_v1
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
secret_name = "Voice"
project_id = "speech-1609595980955"
req = {"name": f"projects/{project_id}/secrets/{secret_name}/versions/latest"}
response = client.access_secret_version(req)
secret = response.payload.data.decode("UTF-8")
s = json.loads(secret)

filepath = "/home/asheeshg01/input/"     #Input audio file path
filepath1 = "/home/asheeshg01/"          
output_filepath = "/home/asheeshg01/" #Final transcript path
#bucketname = "audiofiles_ash" #Name of the bucket created in the step before
bucketname = s.get("bucketname") #Name of the bucket created in the step before

app = Flask(__name__) #creates the Flask instance

# In[70]:

def translate_main(audio_file_name):
    transcript = google_transcribe(audio_file_name)
    transcript_filename = audio_file_name.split('.')[0] + '.txt'
    write_transcripts(transcript_filename,transcript)
    word_details = google_word_details(audio_file_name)
    word_details_filename = transcript_filename.split('.')[0] + '_word_details' + '.txt'
    write_word_details(word_details_filename,word_details)
    sentiment = analyze_sentiment(transcript_filename)
    sentiment_filename = transcript_filename.split('.')[0] + '_sentiment' + '.txt'
    write_sentiment(sentiment_filename,sentiment)

def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")


# In[71]:


def frame_rate_channel(audio_file_name):
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        return frame_rate,channels


# In[72]:


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


# In[73]:


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

# pass filename
ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/app/v1/file-upload', methods=['GET','POST'])
def upload_file():
    if request.method == "POST":
        from Project import translate_main
    # check if the post request has the file part
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message' : 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('/home/asheeshg01/input', filename))
            resp = jsonify({'message' : 'File successfully uploaded'})
            q= Queue(connection=conn)
            job = q.enqueue_call(func=translate_main, args=(file.filename,), result_ttl=50000)
            print(job.get_id())
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({'message' : 'Allowed file types is wav'})
            resp.status_code = 400
            return resp



def google_transcribe(audio_file_name):
    file_name = filepath + audio_file_name
   # mp3_to_wav(file_name)

    # The name of the audio file to transcribe
    
    frame_rate, channels = frame_rate_channel(file_name)
    
    if channels > 1:
        stereo_to_mono(file_name)
    
    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name
    
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''
        
    credential_path = "/home/asheeshg01/Speech-f22e193c0063.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=frame_rate,
    language_code='en-US',
    enable_speaker_diarization=True,
    diarization_speaker_count=2)

    # Detects speech in the audio file
    #operation = client.long_running_recognize(config, audio)
    
    operation = client.long_running_recognize(request={"config":config, "audio":audio})
    response = operation.result(timeout=10000)
    result = response.results[-1] #Changed
    words_info = result.alternatives[0].words #Changed
    
    tag=1 #Changed
    speaker="" #Changed

    for word_info in words_info: #Changed
        if word_info.speaker_tag==tag: #Changed
            speaker=speaker+" "+word_info.word #Changed
        else: #Changed
            transcript += "speaker {}: {}".format(tag,speaker) + '\n' #Changed
            tag=word_info.speaker_tag #Changed
            speaker=""+word_info.word #Changed
          
    
    transcript += "speaker {}: {}".format(tag,speaker) #Changed
    #for result in response.results:
        #transcript += result.alternatives[0].transcript
    
    delete_blob(bucket_name, destination_blob_name)
    return transcript


# In[110]:

def google_word_details(audio_file_name):
    file_name = filepath + audio_file_name
    frame_rate, channels = frame_rate_channel(file_name)
    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''
    word_details = ''
    credential_path = "/home/asheeshg01/Speech-f22e193c0063.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=frame_rate,
    language_code='en-US',
    enable_speaker_diarization=True,
    diarization_speaker_count=2,
    enable_word_time_offsets=True)

    # Detects speech in the audio file
    #operation = client.long_running_recognize(config, audio)
    
    operation = client.long_running_recognize(request={"config":config, "audio":audio})
    response = operation.result(timeout=10000)
    result = response.results[-1] #Changed
    words_info = result.alternatives[0].words #Changed
    
    tag=1 #Changed
    speaker="" #Changed

    for word_info in words_info: #Changed
        word = word_info.word
        start_time = word_info.start_time
        end_time = word_info.end_time
        speaker1 = word_info.speaker_tag
        word_details += " Word: {} : start_time: {}: end_time: {}: speaker {}".format(word,start_time.total_seconds(),end_time.total_seconds(),speaker1)
        
    
    delete_blob(bucket_name, destination_blob_name)
    return word_details


# In[75]:


def write_transcripts(transcript_filename,transcript):
    f= open(output_filepath + transcript_filename,"w+")
    f.write(transcript)
    f.close()


# In[76]:


def write_sentiment(sentiment_filename,sentiment):
    f= open(output_filepath + sentiment_filename,"w+")
    f.write(sentiment)
    f.close()


# In[108]:


def write_word_details(word_details_filename,word_details):
    f= open(output_filepath + word_details_filename,"w+")
    f.write(word_details)
    f.close()


# In[84]:


def analyze_sentiment(transcript_filename):
    """
    Analyzing Sentiment in text file stored in Cloud Storage

    Args:
      gcs_content_uri Google Cloud Storage URI where the file content is located.
      e.g. gs://[Your Bucket]/[Path to File]
    """
    credential_path = "/home/asheeshg01/Speech-f22e193c0063.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    client = language_v1.LanguageServiceClient()
    
    file_name = filepath1 + transcript_filename
    
    bucket_name = bucketname
    source_tr_file_name = filepath1 + transcript_filename
    destination_tr_blob_name = transcript_filename
    
    upload_blob(bucket_name, source_tr_file_name, destination_tr_blob_name)
    
    gcs_content_uri = 'gs://' + bucketname + '/' + transcript_filename
   
    # gcs_content_uri = 'gs://cloud-samples-data/language/sentiment-positive.txt'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"gcs_content_uri": gcs_content_uri, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Get overall sentiment of the input document
    
    sentiment_score= "Document sentiment score: {}".format(response.document_sentiment.score)
    sentiment_magnitude= "Document sentiment magnitude: {}".format(response.document_sentiment.magnitude)
    sentiment = "{} and {}".format(sentiment_score,sentiment_magnitude) #Changed
    delete_blob(bucket_name, destination_tr_blob_name)
    return sentiment

@app.route("/app/v1/<job_key>", methods=['GET'])
def get_results(job_key):
    q= Queue(connection=conn)
    job = q.fetch_job(job_key)

    if job.is_finished:
       return "Yey.Please see the results in the directory!", 201        
    else:
        return "Still job is under progress.Please try after sometime", 202

# In[111]:

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)