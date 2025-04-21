import streamlit as st
import joblib
import nltk
import smtplib
import mysql.connector
import matplotlib.pyplot as plt
from email.message import EmailMessage

# —————————————————————————————
# Download tokenizer
# —————————————————————————————
nltk.download("punkt")

# —————————————————————————————
# Configuration
# —————————————————————————————
EMAIL_ADDRESS    = "studentfeedbacks44@gmail.com"
EMAIL_PASSWORD   = "fhwezznkcxmxfxdo"

MODEL_PATH       = "LogisticRegression_All_shots_data_model.pkl"
VECTORIZER_PATH  = "LogisticRegression_All_shots_data_vectorizer.pkl"
CHART_IMAGE_PATH = "chart_prompt.png"

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "Vigi@2004",
    "database": "Showtell",
    "port":     3306
}

WEEK_LABEL = "Week 5"

# —————————————————————————————
# 1) Load model & vectorizer
# —————————————————————————————
def load_model_and_vectorizer():
    try:
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.stop()

# —————————————————————————————
# 2) Sentence prediction
# —————————————————————————————
def predict_sentences(sentences, model, vectorizer):
    tokens = [" ".join(nltk.word_tokenize(s.lower())) for s in sentences]
    X = vectorizer.transform(tokens)
    return model.predict(X)

# —————————————————————————————
# 3) Database
# —————————————————————————————
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        st.error(f"⚠️ DB connection error: {err}")
        return None

def insert_student_data(name, email, title, story, summary, comments):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO student_inputs
              (name, email, title, story,
               total_sentences, show_sentences, tell_sentences,
               reflection, week, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                name, email, title, story,
                summary["total_sentences"],
                summary["show_sentences"],
                summary["tell_sentences"],
                summary["reflection"],
                WEEK_LABEL,
                comments
            )
        )
        conn.commit()
    except mysql.connector.Error as err:
        st.error(f"⚠️ MySQL insert error: {err}")
    finally:
        cursor.close()
        conn.close()

# —————————————————————————————
# 4) Build & send email
# —————————————————————————————
def build_email_body(name, title, summary, feedback_list, reflection, comments):
    # Summary bullets
    lines = [
        f"Dear {name},",
        "",
        "Thank you for submitting your data story titled “" + title + ".”",
        "",
        "Here’s your summary:",
        f"- Total sentences: {summary['total_sentences']}",
        f"- Show sentences: {summary['show_sentences']}",
        f"- Tell sentences: {summary['tell_sentences']}",
        "",
        "Sentence‑by‑sentence feedback:"
    ]
    # Feedback bullets
    for f in feedback_list:
        status = "✅ Agreed" if f["agree"] else "❌ Did NOT agree"
        lines.append(f"- [{f['label']}] {f['sentence']} → {status}")
    lines += [
        "",
        "Your reflection:",
        f"- {reflection or 'No reflection provided.'}",
        "",
        "Your additional comments:",
        f"- {comments or 'No additional comments.'}",
        "",
        "Best regards,",
        "The Data Story Feedback Team"
    ]
    return "\n".join(lines)

def send_feedback_email(to_email, name, title, summary, feedback_list, reflection, comments):
    body = build_email_body(name, title, summary, feedback_list, reflection, comments)
    msg = EmailMessage()
    msg["Subject"] = f"Feedback for Your Data Story: {title}"
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"❌ Email send error: {e}")

# —————————————————————————————
# 5) Page renderers
# —————————————————————————————
def show_input_page():
    st.markdown("### 📊 Data Story Prompt")
    # Always show your chart
    st.image(CHART_IMAGE_PATH, caption="Use this chart to write your data story.")
    st.write("---")

    st.text_input("Enter your name:", key="student_name")
    st.text_input("Enter your email:", key="student_email")
    st.text_input("Enter a title for your data story:", key="story_title")
    st.text_area("Write your data story here:", key="raw_story")

    if st.button("Analyze"):
        if not (st.session_state.student_name
                and st.session_state.student_email
                and st.session_state.story_title
                and st.session_state.raw_story.strip()):
            st.error("Please fill in name, email, title, and story.")
        else:
            st.session_state.page = "results"

def show_results_page():
    st.markdown("## 🧾 Sentence Analysis")
    model, vectorizer = load_model_and_vectorizer()

    sentences   = nltk.sent_tokenize(st.session_state.raw_story)
    predictions = predict_sentences(sentences, model, vectorizer)

    feedback = []
    show_cnt = tell_cnt = 0

    for i, (sent, pred) in enumerate(zip(sentences, predictions)):
        label = "Show" if pred == 0 else "Tell"
        color = "green" if pred == 0 else "red"
        st.markdown(
            f"<span style='color:{color}'><b>{label}:</b> {sent}</span>",
            unsafe_allow_html=True
        )
        agree = st.checkbox("I agree with this label", key=f"agree_{i}")
        feedback.append({"sentence": sent, "label": label, "agree": agree})
        show_cnt += (pred == 0)
        tell_cnt += (pred == 1)

    st.markdown("## 🗣️ Your Comments")
    st.text_area("Any thoughts on the classifications?", key="comments")

    st.markdown("## 📊 Summary")
    st.write(f"- Total sentences: {len(predictions)}")
    st.write(f"- Show sentences: {show_cnt}")
    st.write(f"- Tell sentences: {tell_cnt}")

    # Chart
    fig, ax = plt.subplots()
    ax.bar(["Show", "Tell"], [show_cnt, tell_cnt])
    ax.set_ylabel("Count")
    ax.set_title("Show vs Tell Breakdown")
    st.pyplot(fig)

    if st.button("Next: Reflection & Email"):
        st.session_state.update({
            "feedback_list": feedback,
            "summary": {
                "total_sentences": len(predictions),
                "show_sentences":  show_cnt,
                "tell_sentences":  tell_cnt,
                # reflection added later
            },
            "page": "reflection"
        })

def show_reflection_page():
    st.markdown("### ✍️ Reflection")
    st.text_area("What did you learn from this feedback?", key="reflection")

    if st.button("Submit Feedback & Send Email"):
        # add reflection to summary
        summary = {**st.session_state.summary, "reflection": st.session_state.reflection}
        insert_student_data(
            st.session_state.student_name,
            st.session_state.student_email,
            st.session_state.story_title,
            st.session_state.raw_story,
            summary,
            st.session_state.comments
        )
        send_feedback_email(
            st.session_state.student_email,
            st.session_state.student_name,
            st.session_state.story_title,
            summary,
            st.session_state.feedback_list,
            st.session_state.reflection,
            st.session_state.comments
        )
        st.success("✅ Feedback submitted and email sent!")
        if st.button("Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

# —————————————————————————————
# 6) Main
# —————————————————————————————
def main():
    st.title("✨ Show or Tell Prediction App ✨")
    if "page" not in st.session_state:
        st.session_state.page = "input"

    if st.session_state.page == "input":
        show_input_page()
    elif st.session_state.page == "results":
        show_results_page()
    else:
        show_reflection_page()

if __name__ == "__main__":
    main()
