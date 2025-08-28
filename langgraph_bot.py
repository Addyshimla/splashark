# langgraph_bot.py
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatState(dict):
    pass

from collections import namedtuple

FAQ = namedtuple('FAQ', ['id', 'question', 'answer'])

faqs = [
    FAQ(id=1, question="How do I update my password?", answer="To update your password, go to the profile section on top right corner and click 'Update password'."),
    FAQ(id=2, question="How can I contact support?", answer="You can contact support by emailing support@yourplatform.com."),
    FAQ(id=3, question="How do I buy a product?", answer="To buy a product, go to the shop page and search for the product you want to buy. Then click on 'Buy Now'. Follow the steps to complete payment."),
    FAQ(id=4, question="How do I sell a product?", answer="Go to your profile, click 'Sell', and fill out the product details. After submission, it will be listed after approval."),
    FAQ(id=5, question="How do I book a service?", answer="Find a provider's profile or listing, then click 'Book Service' and choose your preferred time slot."),
    FAQ(id=6, question="What are subscription plans?", answer="Subscription plans give you access to premium content, early drops, or exclusive features based on the plan."),
    FAQ(id=7, question="What is a drop?", answer="A drop is a user post, like a photo or video, that appears in followers' feeds."),
    FAQ(id=8, question="What is a droplet?", answer="A droplet is a short story-style post that disappears after 24 hours."),
    FAQ(id=9, question="What is a reel?", answer="A reel is a short-form video that can be shared to gain more engagement."),
    FAQ(id=10, question="How do I tip another user?", answer="Tap the tip icon on a user's post or profile. Select an amount and confirm payment to send your tip."),
    FAQ(id=11, question="How do I comment on a drop or reel?", answer="Open the post or reel, scroll down to the comments section, type your message, and tap 'Post'."),
    FAQ(id=12, question="What’s the max video length for a reel?", answer="Currently, the max limit is 20 Mb."),
    FAQ(id=13, question="When does a story expire on the platform?", answer="It expires after 24 hours of posting it."),
    FAQ(id=14, question="What is a droplet?", answer="A droplet is a short story-style post that disappears after 24 hours.")
]

def gpt_node(state):
    input_message = state.get("input", "")
    if not isinstance(input_message, str) or not input_message.strip():
        raise ValueError("Input message cannot be empty or non-string.")
    
    faq_text = "\n".join(f"{faq.id}. {faq.question} – {faq.answer}" for faq in faqs)
    system_prompt = (
        "You are a helpful assistant. Use the following FAQs to assist the user when applicable:\n"
        f"{faq_text}"
    )
    user_prompt = input_message
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    reply = response.choices[0].message.content
    return {"output": reply}



def build_graph():
    builder = StateGraph(dict)
    builder.add_node("chat", gpt_node)
    builder.set_entry_point("chat")
    builder.set_finish_point("chat")
    return builder.compile()

