import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title('Which subreddit do you write like?')

page = st.sidebar.selectbox(
'Select a page:',
('About', 'Make a prediction')
)

if page == 'About':
    st.header('This is my Reddit NLP Project.')
    st.write('')
    st.write("This model, a simple logistic regression, is 81% accurate at predicting whether a reddit comment came from r/fantasy or r/scifi. Now let's see which your input is most like. Select 'Make a Prediction' to try it out.")
    st.write('')
    st.write('Thanks for visiting!')
    st.write('Contact info: clh@cholland.me')

if page == 'Make a prediction':
    st.header('Welcome!')
    st.write('r/fantasy and r/scifi are both pretty fun subreddits. Which one do you belong in?')
    st.write('')

    filename1 = 'models/reddit_vectorizer.sav'
    cv = pickle.load(open(filename1, 'rb'))
    filename2 = 'models/reddit_model.sav'
    model = pickle.load(open(filename2, 'rb'))

    user_text = st.text_input('Please input some text:', value="An elf and an alien walked into a bar ...")

    
    comments = pd.DataFrame([user_text],columns = ['comment'])
    comments = comments['comment']
    comments = cv.transform(comments)
    preds = model.predict(comments)
    pred  = preds[0]
    probs = model.predict_proba(comments)[0]*100
    prob  = np.round(np.max(probs),2)
    outpt = f'I am {prob}% confident that your comment belongs in r/{pred}.'
    
    st.write(outpt)
    
    probs_dict = [{'subreddit': 'r/fantasy', '% Probability':probs[0]},
                  {'subreddit': 'r/scifi',   '% Probability':probs[1]}]
    chart_data = pd.DataFrame(probs_dict).set_index(['subreddit'])
    st.bar_chart(chart_data['% Probability']);
    chk = st.radio('Did I get it right? ', ['No', 'Yes'])
    
    if chk=='Yes':
        st.write('Great! Thanks for trying this out, and feel free to try another comment.')
        st.balloons()
    else:
        st.write("I'm sorry - I'm still learning. Thank you for your feedback.")
   