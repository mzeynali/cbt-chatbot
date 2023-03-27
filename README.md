# CBT ChatBot
This is a chatbot game that helps the teenager to identify his negative thoughts and the therapy tries to move his thoughts towards positive thoughts. According to the conversations that take place between the therapist and the teenager, after several steps, he can offer him a game.


## Requirements

```
pip3 install -r requirements.txt
```

## How To Run
Set your openai api key in line 45 in app.py :  **os.environ["OPENAI_API_KEY"] = "Your_API-KEY"**

```
streamlit run app.py
```

Then enter the below url in your browser.


```
http://Your_IP_Address:8501/
```

The teenager write the answer corresponding to therapist question then press enter ,So if message **not related** not appear, then click on **New Question**
