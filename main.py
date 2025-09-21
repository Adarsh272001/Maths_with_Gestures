import os
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Set Streamlit page config
st.set_page_config(layout="wide")

# Load your custom banner image (assuming 'Math_With_Gestures.png' is in the same directory)
st.image('Math_With_Gestures.png')

# Create Streamlit columns for layout
col1, col2 = st.columns([2, 1])

# Column 1: Camera Feed and Run/Stop checkbox
with col1:
    st.subheader('Live Drawing Area')
    run = st.checkbox('Run Application', value=True)
    FRAME_WINDOW = st.image([])  # Placeholder for the live camera feed

# Column 2: AI Answer Display
with col2:
    st.title('AI Answer')
    # Initialize a Streamlit empty text element to update dynamically
    ai_answer_placeholder = st.empty()
    ai_answer_placeholder.write("Perform the gesture to get an answer...")  # Initial message

load_dotenv()

# Configure the API key and initialize the model object
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set a wider resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set a higher resolution

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    """Detects hands and returns fingers up and landmark list."""
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None  # Return None if no hand is found


def draw(info, prev_pos, canvas, img_for_reset):  # Renamed 'img' to 'img_for_reset' to avoid confusion
    """Draws on the canvas or clears it based on hand gestures."""
    fingers, lmlist = info
    current_pos = None

    # Drawing mode: Index finger up
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmlist[8][0:2]  # Get coordinates of the index fingertip
        if prev_pos is not None:  # Only draw if there's a previous point
            cv2.line(canvas, prev_pos, current_pos, color=(255, 0, 255), thickness=10)

    # Clear canvas mode: All fingers up (thumb, index, middle, ring, pinky)
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img_for_reset)  # Create a new blank canvas
        st.session_state[
            'ai_response'] = "Canvas Cleared. Perform gesture to get answer."  # Clear AI response on screen
        current_pos = None  # Stop drawing after clearing

    return current_pos, canvas


# No longer returns the response, directly updates Streamlit
def sendToAi(model_obj, canvas, fingers):
    """Sends the canvas to the AI if the specific gesture is made."""
    # Gesture to trigger AI: Thumb, Index, and Middle fingers up.
    # This prevents continuous API calls.
    if fingers == [1, 1, 1, 0, 0] and not st.session_state.get('ai_request_sent', False):
        st.session_state['ai_request_sent'] = True  # Set flag to prevent resending

        # FIX: Convert image from BGR (OpenCV) to RGB (PIL/AI Model)
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_canvas)

        print("Sending to AI...")
        ai_answer_placeholder.write("Thinking...")  # Show a loading message
        try:
            response = model_obj.generate_content(["Solve this math problem drawn on the image", pil_image])
            st.session_state['ai_response'] = response.text  # Store response in session state
            print("AI Response:", response.text)
        except Exception as e:
            st.session_state['ai_response'] = f"Error: {e}"  # Store error
            print(f"An error occurred: {e}")

        # Reset the request sent flag after processing
        st.session_state['ai_request_sent'] = False


# Initialize Streamlit session state variables
if 'ai_response' not in st.session_state:
    st.session_state['ai_response'] = "Perform the gesture to get an answer..."
if 'ai_request_sent' not in st.session_state:
    st.session_state['ai_request_sent'] = False

prev_pos = None
canvas = None

# Main application loop
while run:  # Loop runs only if 'run' checkbox is checked
    success, img = cap.read()
    if not success:
        st.error("Failed to read from webcam.")
        break
    img = cv2.flip(img, 1)  # Flip for mirror effect

    # Initialize the canvas on the first frame or after clearing
    if canvas is None:
        canvas = np.zeros_like(img)

    hand_info = getHandInfo(img)
    if hand_info:
        fingers, lmList = hand_info

        # The draw function now returns the updated canvas and position
        prev_pos, canvas = draw(hand_info, prev_pos, canvas, img)

        # Send to AI only when the specific gesture is made
        sendToAi(model, canvas, fingers)  # This function now updates Streamlit directly

    # Combine the camera feed and the canvas
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, gamma=0)

    # Display the combined image in the Streamlit FRAME_WINDOW placeholder
    FRAME_WINDOW.image(image_combined, channels='BGR')

    # Update the AI answer display from session state
    ai_answer_placeholder.write(st.session_state['ai_response'])

    # Streamlit doesn't use cv2.waitKey for its UI loop, it re-runs the script.
    # The 'run' checkbox controls the loop.
else:
    # This block executes when the 'run' checkbox is unchecked
    st.write("Application stopped. Uncheck 'Run Application' to pause.")
    cap.release()  # Ensure camera is released when stopped

cap.release()  # Release camera if loop breaks for other reasons
cv2.destroyAllWindows()  # Not strictly necessary for Streamlit, but good practice