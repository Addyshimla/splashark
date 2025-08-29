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
    FAQ(id=12, question="What's the max video length for a reel?", answer="Currently, the max limit is 20 Mb."),
    FAQ(id=13, question="When does a story expire on the platform?", answer="It expires after 24 hours of posting it."),
    FAQ(id=14, question="What is a droplet?", answer="A droplet is a short story-style post that disappears after 24 hours.")
]

def router_node(state):
    """Router node that decides which path to take"""
    input_message = state.get("input", "")
    if not isinstance(input_message, str) or not input_message.strip():
        raise ValueError("Input message cannot be empty or non-string.")
    
    # Check if user wants image generation
    image_keywords = ["image", "picture", "photo", "generate", "create", "draw", "make", "post"]
    text_lower = input_message.lower()
    
    # Check for image requests
    is_image_request = any(keyword in text_lower for keyword in image_keywords)
    
    return {"route": "image" if is_image_request else "chat", "input": input_message}

def gpt_node(state):
    """Handle regular chat using GPT with FAQ context"""
    input_message = state.get("input", "")
    
    faq_text = "\n".join(f"{faq.id}. {faq.question} – {faq.answer}" for faq in faqs)
    system_prompt = (
        "You are a helpful assistant for a social platform. Use the following FAQs to assist the user when applicable:\n"
        f"{faq_text}\n\n"
        "Answer user questions about the platform features, help with account issues, and provide general assistance."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": input_message}
            ]
        )
        reply = response.choices[0].message.content
        return {"output": reply}
    except Exception as e:
        return {"output": f"Error in chat: {str(e)}"}

def enhance_prompt_node(state):
    """Enhance the user's image prompt for better results"""
    input_message = state.get("input", "")
    
    enhancement_prompt = f"""
    You are a prompt enhancement specialist for DALL-E 3. Take the user's basic image request and enhance it to create a detailed, specific prompt that will generate high-quality images.

    Add relevant details for:
    - Art style (photorealistic, digital art, oil painting, etc.)
    - Lighting (natural, dramatic, soft, golden hour, etc.)  
    - Composition (close-up, wide shot, centered, etc.)
    - Quality descriptors (highly detailed, sharp focus, professional photography, etc.)
    - Mood and atmosphere
    - Colors and textures

    User's request: {input_message}

    Enhanced prompt (respond with ONLY the enhanced prompt, no explanation):
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": enhancement_prompt}]
        )
        enhanced_prompt = response.choices[0].message.content
        return {"enhanced_prompt": enhanced_prompt, "input": input_message}
    except Exception as e:
        # Fallback to original prompt if enhancement fails
        return {"enhanced_prompt": input_message, "input": input_message}

def image_gen_node(state):
    """Handle image generation using DALL-E with enhanced prompt"""
    enhanced_prompt = state.get("enhanced_prompt", state.get("input", ""))
    
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="hd",  # Changed to HD for better quality
            n=1
        )
        
        image_url = response.data[0].url
        return {"output": f"IMAGE_URL:{image_url}"}
    except Exception as e:
        return {"output": f"Error generating image: {str(e)}"}

def build_graph():
    builder = StateGraph(dict)

    builder.add_node("router", router_node)
    builder.add_node("chat", gpt_node)
    builder.add_node("enhance_prompt", enhance_prompt_node)
    builder.add_node("image_gen", image_gen_node)

    builder.set_entry_point("router")

    def route_decision(state):
        return state.get("route", "chat")

    builder.add_conditional_edges(
        "router",
        route_decision,
        {"chat": "chat", "image": "enhance_prompt"}
    )

    builder.add_edge("chat", END)
    
    builder.add_edge("enhance_prompt", "image_gen")
    builder.add_edge("image_gen", END)

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    
    # Test with image request
    result = graph.invoke({"input": "create a image of dog"})
    print(result["output"])
    
    # Test with regular chat
    result = graph.invoke({"input": "How do I update my password?"})
    print(result["output"])