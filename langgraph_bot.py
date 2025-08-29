from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing_extensions import TypedDict
from typing import List, Optional

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define proper TypedDict for state management
class ChatState(TypedDict):
    input: str
    device_type: Optional[str]
    action: Optional[str]
    edit_data: Optional[dict]
    route: Optional[str]
    enhanced_prompt: Optional[str]
    image_url: Optional[str]
    caption: Optional[str]
    hashtags: Optional[List[str]]
    chat_output: Optional[str]
    output: Optional[dict]
    error: Optional[str]

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

def router_node(state: ChatState) -> ChatState:
    """Router node that decides which path to take"""
    print("="*50)
    print("ROUTER NODE DEBUG:")
    print(f"Full state received: {state}")
    
    input_message = state.get("input", "")
    action = state.get("action", "chat")
    
    print(f"Input message: '{input_message}'")
    print(f"Action: {action}")
    print("="*50)
    
    if not isinstance(input_message, str) or not input_message.strip():
        raise ValueError("Input message cannot be empty or non-string")
    
    # Handle different actions
    if action == "regenerate":
        route = "image"
    elif action == "edit_caption":
        route = "edit_caption_only"
    elif action == "edit_hashtags":
        route = "edit_hashtags_only"
    else:
        # Determine route based on keywords for regular chat
        text_lower = input_message.lower().strip()
        image_keywords = ["image", "picture", "photo", "generate", "create", "draw", "make", "post"]
        wants_image = any(keyword in text_lower for keyword in image_keywords)
        route = "image" if wants_image else "chat"
    
    print(f"Routing to: {route}")
    
    return {
        **state,
        "route": route
    }

def gpt_node(state: ChatState) -> ChatState:
    """Handle chat responses using GPT"""
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
        
        return {
            **state,
            "output": reply
        }
    except Exception as e:
        return {
            **state,
            "output": f"Error in chat: {str(e)}"
        }

def enhance_prompt_node(state: ChatState) -> ChatState:
    """Enhance the user's prompt for better image generation"""
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
        
        return {
            **state,
            "enhanced_prompt": enhanced_prompt
        }
    except Exception as e:
        return {
            **state,
            "enhanced_prompt": input_message,
            "error": f"Error enhancing prompt: {str(e)}"
        }

def image_gen_node(state: ChatState) -> ChatState:
    """Generate image using DALL-E 3"""
    enhanced_prompt = state.get("enhanced_prompt", state.get("input", ""))
    
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="hd",
            n=1
        )

        image_url = response.data[0].url
        
        return {
            **state,
            "image_url": image_url
        }
    except Exception as e:
        return {
            **state,
            "image_url": None,
            "error": f"Error generating image: {str(e)}"
        }

def caption_hashtag_node(state: ChatState) -> ChatState:
    """Generate caption and hashtags for the image"""
    enhanced_prompt = state.get("enhanced_prompt", state.get("input", ""))

    system_prompt = """
    You are a social media caption and hashtag generator.
    Based on the user's image description, create:
    - A short, catchy caption
    - 5-7 trending hashtags
    
    Respond strictly in JSON format:
    {
      "caption": "Your catchy caption here",
      "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ]
        )
        
        caption_data = json.loads(response.choices[0].message.content)
        
        return {
            **state,
            "caption": caption_data.get("caption", ""),
            "hashtags": caption_data.get("hashtags", [])
        }
    except Exception as e:
        return {
            **state,
            "caption": "Check out this awesome image!",
            "hashtags": ["#ai", "#generated", "#cool"],
            "error": f"Error generating caption: {str(e)}"
        }

def final_output_node(state: ChatState) -> ChatState:
    """Prepare the final output based on the route taken"""
    route = state.get("route", "chat")
    
    if route == "chat":
        # For chat responses, output is already set by gpt_node
        return state
    elif route == "image":
        # For image responses, combine all image-related data
        result = {}
        
        if state.get("image_url"):
            result["image_url"] = state["image_url"]
            
        if state.get("caption"):
            result["caption"] = state["caption"]
            
        if state.get("hashtags"):
            result["hashtags"] = state["hashtags"]
            
        if state.get("error"):
            result["error"] = state["error"]
            
        # If no image was generated successfully, provide error message
        if not result.get("image_url"):
            result = "Sorry, I couldn't generate the image. Please try again with a different prompt."
        
        return {
            **state,
            "output": result
        }
    
    return state

def build_graph():
    """Build the LangGraph workflow"""
    builder = StateGraph(ChatState)

    # Add all nodes
    builder.add_node("router", router_node)
    builder.add_node("chat", gpt_node)
    builder.add_node("enhance_prompt", enhance_prompt_node)
    builder.add_node("image_gen", image_gen_node)
    builder.add_node("caption_hashtag", caption_hashtag_node)
    builder.add_node("final_output", final_output_node)

    # Set entry point
    builder.set_entry_point("router")

    # Define routing logic
    def route_decision(state: ChatState) -> str:
        route = state.get("route", "chat")
        print(f"Route decision: {route}")
        return route

    # Add conditional edges from router
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat": "chat",
            "image": "enhance_prompt"
        }
    )

    # Chat flow - simple path to end
    builder.add_edge("chat", "final_output")

    # Image flow - enhanced prompt -> image generation -> caption -> final output
    builder.add_edge("enhance_prompt", "image_gen")
    builder.add_edge("image_gen", "caption_hashtag")
    builder.add_edge("caption_hashtag", "final_output")

    # All paths end at final_output, then END
    builder.add_edge("final_output", END)

    return builder.compile()

