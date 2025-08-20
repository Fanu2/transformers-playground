import streamlit as st
import torch
from transformers import pipeline

st.set_page_config(page_title="Transformers Playground", page_icon="🤗", layout="centered")

st.title("🤗 Transformers Playground")
st.write("Test Hugging Face Transformers pipelines for **text generation** and **chat**.")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
task = st.sidebar.selectbox("Choose task", ["Text Generation", "Chat"])
if task == "Text Generation":
    default_model = "Qwen/Qwen2.5-1.5B"
else:
    default_model = "meta-llama/Meta-Llama-3-8B-Instruct"

model_name = st.sidebar.text_input("Model name", value=default_model)

# Load pipeline lazily (to avoid long startup time)
@st.cache_resource
def load_pipeline(task, model_name):
    if task == "Text Generation":
        return pipeline(task="text-generation", model=model_name)
    else:
        return pipeline(
            task="text-generation", 
            model=model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None
        )

pipe = load_pipeline(task, model_name)

# === Text Generation ===
if task == "Text Generation":
    prompt = st.text_area("✍️ Enter your prompt", "The secret to baking a really good cake is ")
    max_tokens = st.slider("Max new tokens", 20, 512, 128)

    if st.button("Generate"):
        with st.spinner("Generating..."):
            output = pipe(prompt, max_new_tokens=max_tokens)
            st.success(output[0]["generated_text"])

# === Chat Mode ===
else:
    st.write("💬 Chat with your model (system + user messages).")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."}
        ]

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("system").write(msg["content"])

    user_input = st.chat_input("Type your message")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = pipe(st.session_state.chat_history, max_new_tokens=256)
                # Grab last message content
                reply = response[0]["generated_text"][-1]["content"]
                st.write(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
