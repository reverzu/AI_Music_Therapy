import streamlit as st
from streamlit_elements import elements, mui, html, sync
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from llama_cpp import Llama
import os
import random
import pandas as pd
import base64
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit_player import st_player
import polarplot
import songrecommendations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr


SPOTIPY_CLIENT_ID = 'YOUR_SPOTIFY_CLIENT_ID'
SPOTIPY_CLIENT_SECRET = 'YOUR_SPOTIFY_CLIENT_SECRET'

auth_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)


# Loading llama2 chat model
# llm = Llama(model_path="C:/Users/abhij/AppData/Local/llama_index/models/llama-2-7b.ggmlv3.q4_K_M.bin")
llm = Llama(
    model_path="./models/llama-2-7b-chat.ggmlv3.q8_0.bin")


# Loading Model for Emotion Detection
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Setting RUN session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"


try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""


if not (emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        ##############################
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50),
                        cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(
                                   color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(
            frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(
            frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


st.set_page_config(page_title="🎶 AI Music Therapy", layout='wide')


st.markdown("""
<style>
.block-container{
padding-top:1.5rem;
}
			.reportview-container {
        background: url("C:/llama2.1/images/background/music_notes.jpg")
    }

.container {
  display: inline-block;
}		
.typed-out{
    overflow: hidden;
    border-right: .15em solid orange;
    white-space: nowrap;
    font-size: 1.1rem;
	margin: 0 auto; /* Gives that scrolling effect as the typing happens */
	letter-spacing: .15em; /* Adjust as needed */
    width: 0;
    animation: typing 1s forwards,
	blink-caret .75s step-end infinite;
            font-weight: 550;
}		
.typewriter{
overflow: hidden; /* Ensures the content is not revealed until the animation */
border-right: .12em solid orange; /* The typwriter cursor */
white-space: nowrap; /* Keeps the content on a single line */
font-size: 1.6rem;
margin: 0 auto; /* Gives that scrolling effect as the typing happens */
letter-spacing: .15em; /* Adjust as needed */
animation: 
typing 3.5s forwards,
blink-caret .75s step-end infinite;
font-size:1rem;
display: inline-block;
width: 0;
}

/* The typing effect */
@keyframes typing {
from { width: 0 }
to { width: 100% }
}

/* The typewriter cursor effect */
@keyframes blink-caret {
from, to { border-color: transparent }
50% { border-color: orange; }
}
</style>
""", unsafe_allow_html=True)


st.sidebar.header("🎶 AI Music Therapy")
st.sidebar.title("Main Menu")
search_choices = ['Home', 'Song/Track', 'Artist', 'Album', 'Research']
search_selected = st.sidebar.selectbox(
    "Your search choice please: ", search_choices)
search_results = []
tracks = []
artists = []
albums = []

music_quotes = [
    """Music expresses that which cannot be said and on which it is impossible to be silent." - Victor Hugo""",
    """Music washes away from the soul the dust of everyday life." - Berthold Auerbach""",
    """Without music, life would be a mistake." - Friedrich Nietzsche""",
    """Music is the strongest form of magic." - Franz Liszt""",
    """If music be the food of love, play on." - William Shakespeare""",
    """After silence, that which comes nearest to expressing the inexpressible is music." - Aldous Huxley""",
    """The cool thing about music is that no one can take music away from you, writing-wise." - Ray Charles""",
    """Where words fail, music speaks." - Hans Christian Andersen"""
]

if search_selected == "Home":

    st.title("🎶 AI Music Therapy")
    st.markdown(
	f'<div class="container"><div class="typed-out">"{music_quotes[random.randint(0, len(music_quotes) - 1)]}</div></div><br><br>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(['Home', 'Chat Companion', "AI Music Generator"])
    with tab1:
        col_main1, col_main2 = st.columns([8, 4])
        with col_main1:
            articles_dict = {
            """Music's Magical Melody: How Music Therapy Soothes the Soul""": ["""This article explores the scientific underpinnings of music therapy, delving into how music's rhythm, melody, and harmony impact our brain and emotions. It showcases real-life stories of individuals who found solace and healing through music therapy.""", f'https://us.sagepub.com/en-us/nam/node/7006/print'],
            """From Nursery Rhymes to Rehabs: The Diverse Applications of Music Therapy""": ["""This piece tackles the misconception that music therapy is just for relaxation. It highlights its diverse applications, from helping children with developmental delays to aiding in pain management for patients undergoing surgery.""", f'https://www.musictherapy.org/'],
            """Unleashing the Inner Musician: Music Therapy as a Tool for Self-Expression""": ["""This article focuses on the empowering aspect of music therapy, showcasing how it can unlock hidden potential and provide a safe space for individuals to express themselves creatively. It explores methods like songwriting and improvisation as therapeutic tools.""", f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9286888/'],
            """Beyond Notes and Chords: The Healing Power of Music Listening""": ["""This piece expands the scope of music therapy beyond active participation. It explores the therapeutic benefits of simply listening to music, highlighting curated playlists and specific genres that can alleviate anxiety, enhance mood, and improve cognitive function.""", f' https://www.verywellmind.com/'],
            """Music Therapy: A Symphony of Hope for Chronic Conditions""": ["""This article sheds light on the role of music therapy in managing chronic conditions like pain, fatigue, and depression. It showcases how tailored music interventions can improve quality of life for individuals facing long-term health challenges.""", f'https://www.musictherapy.org/about/musictherapy/']
		}
            key, value = random.choice(list(articles_dict.items()))
            st.subheader(key)
            st.image("./images/slides/" +
                 random.choice(os.listdir("./images/slides/")))
            st.markdown(value[0])
            st.markdown(value[1])
            st.markdown('---')
            mainHeaderPlaceholder = st.empty()
            mainHeader = mainHeaderPlaceholder.header(
            "Hello there, How are we feeling today?", divider='rainbow')

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                happyPlaceholder = st.empty()
                happy = happyPlaceholder.button(
                "Play Something Happy 😊", use_container_width=True)
            with col2:
                neutralPlaceholder = st.empty()
                neutral = st.button("Play Something Netural 😐", use_container_width=True)
            with col3:
                sadPlaceholder = st.empty()
                sad = st.button("Play Something Sad 😢"
                               , use_container_width=True)

            def songOnEmotion(emotion):
                if emotion == 'happy':
                    song = random.choice(os.listdir("./music/happy/"))
                    # audio = open("./music/happy/"+song)
                    return ("./music/happy/"+song)
                    # happyPlaceholder.empty()
                elif emotion == 'neutral':
                    directories = ["./music/neutral/",
                               "./music/happy/", "./music/sad/"]
                    random_dir_index = random.randint(0, len(directories) - 1)
                    chosen_directory = directories[random_dir_index]
                    song = random.choice(os.listdir(chosen_directory))
                    # audio = open("./music/happy/"+song)
                    return (chosen_directory+song)
                    # neutralPlaceholder.empty()
                elif emotion == 'sad':
                    song = random.choice(os.listdir("./music/sad/"))
                    # audio = open("./music/happy/"+song)
                    return ("./music/sad/"+song)

                    # sadPlaceholder.empty()

            def imageOnEmotion(emotion):
                if emotion == 'happy':
                    song = random.choice(os.listdir("./images/happy/"))
                    # audio = open("./music/happy/"+song)
                    return ("./images/happy/"+song)
                    # happyPlaceholder.empty()
                elif emotion == 'neutral':
                    directories = ["./images/neutral/",
                               "./images/happy/", "./images/sad/"]
                    random_dir_index = random.randint(0, len(directories) - 1)
                    chosen_directory = directories[random_dir_index]
                    song = random.choice(os.listdir(chosen_directory))
                    # audio = open("./music/happy/"+song)
                    return (chosen_directory+song)
                    # neutralPlaceholder.empty()
                elif emotion == 'sad':
                    song = random.choice(os.listdir("./images/sad/"))
                    # audio = open("./music/happy/"+song)
                    return ("./images/sad/"+song)

                    # sadPlaceholder.empty()

            if happy:
                st.image(imageOnEmotion('happy'))
                st.audio(songOnEmotion('happy'))
            elif neutral:
                st.image(imageOnEmotion('neutral'))
                st.audio(songOnEmotion('neutral'))
            elif sad:
                st.image(imageOnEmotion('sad'))
                st.audio(songOnEmotion('sad'))

            st.markdown('----')
            st.text('')
            container = st.container()
            container.subheader(
                "Capture Emotion from your webcam?", divider='rainbow')

            colb1, colb2 = st.columns([1, 1])
            with colb1:
                emotionCaptureButtonPlaceholder = st.empty()
                emotionCaptureButton = emotionCaptureButtonPlaceholder.button(
                    "Capture My Facial Emotion 📷", use_container_width=True)
            with colb2:
                PlaySomethingElseButtonPlaceholder = st.empty()
                PlaySomethingElseButton = PlaySomethingElseButtonPlaceholder.button(
                    "Play Something Else 🎵", use_container_width=True)
            if st.session_state["run"] != "false":
                webrtc_streamer(key="key", desired_playing_state=True,
                            video_processor_factory=EmotionProcessor)
            container.text("")
            st.markdown('---')
            st.text("")
            st.subheader(
                "Analyze Sentiment from your Voice?", divider='rainbow')
            colc1, colc2 = st.columns([1, 1])
            with colc1:
                voiceSentimentAnalyzerButtonPlaceholder = st.empty()
                voiceSentimentAnalyzerButton = voiceSentimentAnalyzerButtonPlaceholder.button("Record my Voice 🗣🎙", use_container_width=True)
            with colc2:
                PlaySomethingElseButton2Placeholder = st.empty()
                PlaySomethingElseButton2 = PlaySomethingElseButton2Placeholder.button(
                    "Play Something Else 🎵 ", use_container_width=True)
                
            # wav_audio_data = st_audiorec()
            # if wav_audio_data is not None:
            #     st.audio(wav_audio_data, format='audio/wav')
            #     recognizer = sr.Recognizer()
            #     try:
            #         print("Printing the Message: ")
            #         text = recognizer.recognize_google(wav_audio_data, language='en-US')
            #         st.text(f'Your Message:{text}')
            #     except Exception as ex:
            #         st.error(ex)


            if emotionCaptureButton:
                if not (emotion):
                    container.warning("Please let me capture your emotion first")
                    st.session_state["run"] = "true"
                elif emotion:
                    container.subheader("It looks like you're " + emotion +
                                    "\nLet me play some " + emotion + " Music for you")
                    container.image(imageOnEmotion(emotion))
                    container.audio(songOnEmotion(emotion))
                    np.save("emotion.npy", np.array([""]))
                    st.session_state["run"] = "false"
                    emotion = np.array([""])[0]

            elif PlaySomethingElseButton:
                container.subheader("It looks like you're " + emotion +
                                "\nLet me play some " + emotion + " Music for you")
                container.image(imageOnEmotion(emotion))
                container.audio(songOnEmotion(emotion))

            # st.markdown('----')
            # feeling = st.text_input("Feeling: ", placeholder="How do you feel?", key="feel")
                

            if voiceSentimentAnalyzerButton:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.text("Clearing Background Noises...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    st.text("Waiting for you message...")
                    recordedAudio = recognizer.listen(source)
                    st.text("Recording Done")
                try:
                    text = recognizer.recognize_google(recordedAudio, language='en-US')
                except Exception as ex:
                    st.error(ex)
                sentence = [str(text)]
                analyzer = SentimentIntensityAnalyzer()

                for i in sentence:  
                    ps = analyzer.polarity_scores(i)
                    # st.text(ps)
                if ps:
                    compound = float(ps.get('compound'))
                    if float(compound)>=-0.35 and float(compound)<=0.35:
                        emotion = 'neutral'
                        np.save("emotion.npy", np.array(['neutral']))
                    elif float(compound) < -0.35:
                        emotion='sad'
                        np.save("emotion.npy", np.array(['sad']))
                    elif float(compound) > 0.35:
                        emotion='happy'
                        np.save("emotion.npy", np.array(['happy']))
                st.subheader("Your audio feels like you're " + emotion +
                                    "\nLet me play some " + emotion + " Music for you")
                st.image(imageOnEmotion(emotion))
                st.audio(songOnEmotion(emotion))
            elif PlaySomethingElseButton2:
                st.subheader("Your audio feels like you're " + emotion +
                                    "\nLet me play some " + emotion + " Music for you")
                st.image(imageOnEmotion(emotion))
                st.audio(songOnEmotion(emotion))


        def template(context, question):
            return f"""Use the following pieces of information to answer the user's question.
	If you don't know the answer, just say that you don't know, don't try to make up an answer.
	Context: {context}
	Question: {question}
	Only return the helpful answer below and nothing else.
	Helpful answer:
	"""
        pre_prompt = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nGenerate the response by answering the question as succinctly as possible. If you cannot answer the question, please state that you do not have an answer.\n"""

        def prompter(context, question):
            prompt = pre_prompt + f"CONTEXT:\n\n{context}\n" + \
            f"Question : {question}" + "[\INST]"
            return prompt

        with col_main2:
            st.markdown(f'**Related Videos**')
            st_player('<iframe width="866" height="487" src="https://www.youtube.com/embed/la-2vEPLTyk" title="Steve Parker Artist Talk | FIGHT SONG" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>')
            st_player('<iframe width="702" height="396" src="https://www.youtube.com/embed/Rk44ml9WFew" title="Song of the Ambassadors at Lincoln Center" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>')
            st_player('<iframe width="866" height="487" src="https://www.youtube.com/embed/UfcAVejslrU" title="Marconi Union - Weightless (Official Video)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>')
            st_player('<iframe width="866" height="487" src="https://www.youtube.com/embed/6ngfA9y6eUU" title="Digitonal - Mirtazapine (The Ambient Zone)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>')
            st_player('<iframe width="866" height="487" src="https://www.youtube.com/embed/uTTmD2sAzQY" title="Chris Coco - Waterfall (The Ambient Zone)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>')
            # st.video('./videos/Steve Parker Artist Talk _ FIGHT SONG.mp4')
            # st.video('./videos/Song of the Ambassadors at Lincoln Center.mp4')
            # st.video('./videos/Marconi Union - Weightless (Official Video).mp4')
            # st.video('./videos/Digitonal - Mirtazapine (The Ambient Zone).mp4')
            # st.video('./videos/Chris Coco - Waterfall (The Ambient Zone).mp4')
            
        
    with tab2:
            chats_dict = {
                "What is AI Music Therapy?": "AI Music Therapy is a form of therapy that utilizes artificial intelligence (AI) to create personalized music experiences for individuals with mental health conditions or disabilities. It involves the use of machine learning algorithms and natural language processing to analyze a person's emotional state and generate customized music tracks that can help them relax, reduce anxiety, or improve their mood. AI Music Therapy is still a relatively new field, but it has shown promising results in early studies and has the potential to provide an innovative and effective approach to mental health treatment.",
                "What is Music Therapy?" : """Music therapy is a form of therapy that uses music to help individuals cope with various challenges and issues in their lives. It involves working with a trained music therapist who uses music and music-making techniques to promote emotional healing, social skills development, and cognitive function improvement. Music therapy can be used to treat a wide range of conditions, including mental health disorders, neurological impairments, and physical disabilities. It can also be used to enhance personal growth and well-being.

Music therapy is based on the belief that music has the power to elicit deep emotional responses and promote healing at a profound level. By working with music, individuals can gain insight into their thoughts, feelings, and behaviors, and develop new coping skills and strategies for managing challenges in their lives.

Some of the key techniques used in music therapy include:

Improvisation: creating music spontaneously in the moment to express emotions and experiences
Composition: writing and recording original songs to express personal experiences and emotions
Music listening: actively engaging with and responding to pre-recorded music to explore emotions and gain insight into one's self
Music movement: using music to facilitate physical movements and gestures that can help to release tension, improve coordination, and enhance mood.
Music therapy is a highly individualized form of therapy, and the specific techniques used will depend on the taste of the individual""",
                "Is Music Therapy useful?": '''Yes, music therapy has been shown to be a useful tool in improving mental health and well-being. Research has consistently demonstrated its benefits in reducing stress, anxiety, and depression, as well as improving cognitive function and social skills. Music therapy can be particularly helpful for individuals with neurological or psychiatric disorders, such as Alzheimer's disease, dementia, post-traumatic stress disorder (PTSD), and schizophrenia. Additionally, music therapy has been used to help individuals cope with grief, loss, and trauma.

However, it is important to note that music therapy should be provided by a licensed and trained professional, as improper use of music can have unintended negative consequences. For example, loud or disruptive music can exacerbate anxiety or stress, while inappropriate lyrics can cause emotional distress.

Overall, music therapy has the potential to be a valuable addition to traditional therapeutic approaches, and its benefits should not be underestimated.'''
			}
            col_main21, gap, col_main22 = st.columns([8, 0.1, 4])
            with col_main21:
                output = ''
                st.subheader("🎶 AI Music Therapy Chat Companion")
                enquiry = st.text_input("Ask us anything related to AI Music Therapy 😊"
                                    ,placeholder="What would you like to know?", key="enquiry")
                colb1, colb2 = st.columns([1, 1])
                load_box = st.empty()
                res_box = st.empty()
                ask = colb1.button("Ask", type='primary', use_container_width=True)
                stop = colb2.button("Stop", use_container_width=True)
                if ask:
                    load_box.text('Generating Response...')
                    resp = []
                    # print(prompter('Music', enquiry))
                    for output in llm(
                        # f"Question: {prompter(emotion, job='student', feeling='depleted')} Answer:",
                        f"Question: {prompter('Music', enquiry)}? Answer:",
                        max_tokens=0,
                        # stop=[" \n", "Question:", "Q:"],
                        stream=True,
                    ):
                        if stop:
                            break
                        resp.append(output["choices"][0]["text"])
                        result = "".join(resp).strip()
                        # result = result.replace("\n", "")
                        res_box.markdown(f"{result}")
                    load_box.empty()
                    st.markdown('----')
                with col_main22:
                    st.markdown(f'<h4 style="padding-bottom:0;">Frequently Asked Questions</h4>', unsafe_allow_html=True)
                    st.markdown('---')
                    for key, value in chats_dict.items():
                        with st.expander(key):
                            st.markdown(f'{value}')
    with tab3:
        tab3_col1, tab3_col2 = st.columns([8, 4])
        with tab3_col2:
            st.subheader("Additional Details")
            music_duration = st.slider("Music Duration", 1, 30, 10)
            sample_rate_different = st.text_input("Enter Sample Rate(in Hz, only int)", placeholder="Default 32000Hz")

            st.markdown('__Some previously generated__')
            for i in os.listdir(r"./aiGeneratedMusic/"):
                audio_file = "./aiGeneratedMusic/" + i
                st.audio(audio_file)

        with tab3_col1:
            music_generator_prompt = st.text_input("What type of Music do you want to listen?", placeholder="Enter some prompt to generate Music")
            generate_music_button = st.button("Generate 🎶", type='primary')
            if music_generator_prompt and generate_music_button:
                # loading = st.empty()
                loading_progress = st.spinner("Generating Music... (Please be patient, it may take a few minutes)")
                with loading_progress:
                    from audiocraft.models import musicgen
                    from audiocraft.utils.notebook import display_audio
                    import torchaudio
                    import os
                    import random
                    st.text("Resources loaded successfully")
                    output_directory = r"./aiGeneratedMusic/"

                    model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
                    model.set_generation_params(duration = int(music_duration) if music_duration else 10)
                    prompted_list = [music_generator_prompt]
                    res = model.generate(prompted_list, progress=True)
                    num = int(random.random()*10**10)
                    for i, audio in enumerate(res):
                        audio_cpu = audio.cpu()
                        file_path = os.path.join(
                            output_directory, f'{prompted_list[i].split(" ")[0]}_audio-{num}.wav')
                        torchaudio.save(file_path, audio_cpu, sample_rate= int(sample_rate_different) if sample_rate_different else 32000)
                    for i in range(len(res)):
                        file_path = os.path.join(
                            output_directory, f'{prompted_list[i].split(" ")[0]}_audio-{num}.wav')
                        # audio, sample_rate = torchaudio.load(file_path)
                        st.image(imageOnEmotion('neutral'))
                        st.audio(file_path)
        




elif search_selected == "Song/Track":
    st.title("🎶 Search any Song/Track")
    search_keyword = st.text_input(search_selected + " (Keyword Search)")
    button_clicked = st.button("Search")
    st.subheader("Start song/track search")
    tracks = sp.search(q='track:' + search_keyword, type='track', limit=20)
    tracks_list = tracks['tracks']['items']
    if len(tracks_list) > 0:
        for track in tracks_list:
            # st.write(track['name'] + " - By - " + track['artists'][0]['name'])
            search_results.append(
                track['name'] + " - By - " + track['artists'][0]['name'])
elif search_selected == "Artist":
    st.title("🎶 Search any Artist")
    search_keyword = st.text_input(search_selected + " (Keyword Search)")
    button_clicked = st.button("Search")
    st.subheader("Start artist search")
    artists = sp.search(q='artist:' + search_keyword, type='artist', limit=20)
    artists_list = artists['artists']['items']
    if len(artists_list) > 0:
        for artist in artists_list:
            # st.write(artist['name'])
            search_results.append(artist['name'])
elif search_selected == "Album":
    st.title("🎶 Search any Album")
    search_keyword = st.text_input(search_selected + " (Keyword Search)")
    button_clicked = st.button("Search")
    st.subheader("Start album search")
    albums = sp.search(q='album:' + search_keyword, type='album', limit=20)
    albums_list = albums['albums']['items']
    if len(albums_list) > 0:
        for album in albums_list:
            # st.write(album['name'] + " - By - " + album['artists'][0]['name'])
            # print("Album ID: " + album['id'] + " / Artist ID - " + album['artists'][0]['id'])
            search_results.append(
                album['name'] + " - By - " + album['artists'][0]['name'])
elif search_selected == "Research":
    def displayPDF(file):
        # Opening file from file path
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf"></iframe>'

    # Displaying File
        st.markdown(pdf_display, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(
        ['Industrial Designers Society of America', 'World Federation of Music Therapy'])
    with tab1:
        displayPDF(
            "./pdf/Artificial-Intelligence-Music-Therapy-and-the-Fight-Against-Mental-Illness.pdf")
    with tab2:
        displayPDF(
            "./pdf/World Federation of Music Therapy.pdf")
        
        

selected_album = None
selected_artist = None
selected_track = None
if search_selected == 'Song/Track':
    selected_track = st.selectbox("Select your song/track: ", search_results)
elif search_selected == 'Artist':
    selected_artist = st.selectbox("Select your artist: ", search_results)
elif search_selected == 'Album':
    selected_album = st.selectbox("Select your album: ", search_results)

if selected_track is not None and len(tracks) > 0:
    tracks_list = tracks['tracks']['items']
    track_id = None
    if len(tracks_list) > 0:
        for track in tracks_list:
            str_temp = track['name'] + " - By - " + track['artists'][0]['name']
            if str_temp == selected_track:
                track_id = track['id']
                track_album = track['album']['name']
                img_album = track['album']['images'][0]['url']
                # st.write(track_id, track_album)
                # st.image(img_album)
                songrecommendations.save_album_image(img_album, track_id)
    selected_track_choice = None
    if track_id is not None:
        image = songrecommendations.get_album_mage(track_id)
        st.image(image)
        track_choices = ['Song Features', 'Similar Songs Recommendation']
        selected_track_choice = st.sidebar.selectbox(
            'Please select track choice: ', track_choices)
        if selected_track_choice == 'Song Features':
            track_features = sp.audio_features(track_id)
            df = pd.DataFrame(track_features, index=[0])
            df_features = df.loc[:, ['acousticness', 'danceability', 'energy',
                                     'instrumentalness', 'liveness', 'speechiness', 'valence']]
            st.dataframe(df_features)
            polarplot.feature_plot(df_features)
        elif selected_track_choice == 'Similar Songs Recommendation':
            token = songrecommendations.get_token(
                SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
            similar_songs_json = songrecommendations.get_track_recommendations(
                track_id, token)
            recommendation_list = similar_songs_json['tracks']
            recommendation_list_df = pd.DataFrame(recommendation_list)
            # st.dataframe(recommendation_list_df)
            recommendation_df = recommendation_list_df[[
                'name', 'explicit', 'duration_ms', 'popularity']]
            st.dataframe(recommendation_df)
            # st.write("Recommendations....")
            songrecommendations.song_recommendation_vis(recommendation_df)

    else:
        st.write("Please select a track from the list")

elif selected_album is not None and len(albums) > 0:
    albums_list = albums['albums']['items']
    album_id = None
    album_uri = None
    album_name = None
    if len(albums_list) > 0:
        for album in albums_list:
            str_temp = album['name'] + " - By - " + album['artists'][0]['name']
            if selected_album == str_temp:
                album_id = album['id']
                album_uri = album['uri']
                album_name = album['name']
    if album_id is not None and album_uri is not None:
        st.markdown(f"**All tracks for the album : {album_name}**")
        # st.markdown(album_name)
        album_tracks = sp.album_tracks(album_id)
        df_album_tracks = pd.DataFrame(album_tracks['items'])
        # st.dataframe(df_album_tracks)
        df_tracks_min = df_album_tracks.loc[:,
                                            ['id', 'name', 'duration_ms', 'explicit', 'preview_url']]
        # st.dataframe(df_tracks_min)
        for idx in df_tracks_min.index:
            with st.container():
                col1, col2, col3, col4 = st.columns((4, 4, 1, 1))
                col11, col12 = st.columns((8, 2))
                col1.write(df_tracks_min['id'][idx])
                col2.write(df_tracks_min['name'][idx])
                col3.write(df_tracks_min['duration_ms'][idx])
                col4.write(df_tracks_min['explicit'][idx])
                if df_tracks_min['preview_url'][idx] is not None:
                    col11.write(df_tracks_min['preview_url'][idx])
                    # with col12:
                    st.audio(df_tracks_min['preview_url']
                             [idx], format="audio/mp3")


if selected_artist is not None and len(artists) > 0:
    artists_list = artists['artists']['items']
    artist_id = None
    artist_uri = None
    selected_artist_choice = None
    if len(artists_list) > 0:
        for artist in artists_list:
            if selected_artist == artist['name']:
                artist_id = artist['id']
                artist_uri = artist['uri']

    if artist_id is not None:
        artist_choice = ['Albums', 'Top Songs']
        selected_artist_choice = st.sidebar.selectbox(
            'Select artist choice', artist_choice)

    if selected_artist_choice is not None:
        if selected_artist_choice == 'Albums':
            artist_uri = 'spotify:artist:' + artist_id
            album_result = sp.artist_albums(artist_uri, album_type='album')
            all_albums = album_result['items']
            col1, col2, col3 = st.columns((6, 4, 2))
            for album in all_albums:
                col1.write(album['name'])
                col2.write(album['release_date'])
                col3.write(album['total_tracks'])
        elif selected_artist_choice == 'Top Songs':
            artist_uri = 'spotify:artist:' + artist_id
            top_songs_result = sp.artist_top_tracks(artist_uri)
            for track in top_songs_result['tracks']:
                with st.container():
                    col1, col2, col3, col4 = st.columns((4, 4, 2, 2))
                    col11, col12 = st.columns((10, 2))
                    col21, col22 = st.columns((11, 1))
                    col31, col32 = st.columns((11, 1))
                    col1.write(track['id'])
                    col2.write(track['name'])
                    if track['preview_url'] is not None:
                        col11.write(track['preview_url'])
                        with col12:
                            st.audio(track['preview_url'], format="audio/mp3")
                    with col3:
                        def feature_requested():
                            track_features = sp.audio_features(track['id'])
                            df = pd.DataFrame(track_features, index=[0])
                            df_features = df.loc[:, [
                                'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
                            with col21:
                                st.dataframe(df_features)
                            with col31:
                                polarplot.feature_plot(df_features)

                        feature_button_state = st.button(
                            'Track Audio Features', key=track['id'], on_click=feature_requested)
                    with col4:
                        def similar_songs_requested():
                            token = songrecommendations.get_token(
                                SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
                            similar_songs_json = songrecommendations.get_track_recommendations(
                                track['id'], token)
                            recommendation_list = similar_songs_json['tracks']
                            recommendation_list_df = pd.DataFrame(
                                recommendation_list)
                            recommendation_df = recommendation_list_df[[
                                'name', 'explicit', 'duration_ms', 'popularity']]
                            with col21:
                                st.dataframe(recommendation_df)
                            with col31:
                                songrecommendations.song_recommendation_vis(
                                    recommendation_df)

                        similar_songs_state = st.button(
                            'Similar Songs', key=track['id']+track['name'], on_click=similar_songs_requested)
                    st.write('----')
