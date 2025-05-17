import os
import time
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini("Recorded Video December 21, 2024 - 9:20AM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 9:36AM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 10:03AM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 10:30AM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 10:32AM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 3:31PM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 3:44PM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 3:53PM.x-matroska", mime_type="video/x-matroska"),
  upload_to_gemini("Recorded Video December 21, 2024 - 4:34PM.x-matroska", mime_type="video/x-matroska"),
]

# Some files have a processing delay. Wait for them to be ready.
wait_for_files_active(files)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        files[0],
      ],
    },
    {
      "role": "user",
      "parts": [
        "tell me everything about this video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Certainly! Here's a breakdown of what's happening in the video:\n\n**Overall Scene:**\n\nThe video appears to be a recording of a video call or a live stream, likely from a conference, competition or other event being held at an educational institution or professional setting. The camera is fixed and focused on two people sitting at a table. \n\n**Key Elements:**\n\n*   **People:** There are two individuals visible.\n    *   **Man:**  He is in the foreground, wearing a black suit jacket and light blue shirt. He appears to be the main focus of the video. He initially speaks and waves at the camera, and then continues talking for the rest of the segment. He has dark hair, a beard and a mustache.\n    *   **Woman:**  She is visible in the left side of the frame. She is also wearing a dark suit jacket. Initially, she is not looking at the camera but then she looks down and slightly to the left. \n\n*  **Setting:**  The background suggests an event space, possibly a classroom or hall. \n    *   **Walls:** The walls are light-colored, with two banners visible. The one on the left says \"Hackathon Gyanotsav 2025\".  The banner on the right mentions something about \"Welcome to ...\". There are also some colorful ribbons hanging as decorations.\n    *   **Furniture:** There are tables and chairs arranged in rows. Some chairs are unoccupied. \n    *   **Lighting:** The room is lit by overhead lights, with a bright highlight visible, suggesting artificial lights.\n    *   **Air Conditioner Vent:** There is an Air conditioning vent visible in the top right of the frame. \n\n*   **Video Quality:** The video quality is slightly grainy.\n\n**Action:**\n\n*   The man starts by facing the camera and speaking.\n*   In the beginning he waves to the camera.\n*   He continues to talk while sitting, while the woman next to him appears to be listening to him.\n\n**Additional Observations**\n\n*   The presence of the \"Hackathon\" banner indicates the video is likely related to a coding or technology event.\n*   The people are dressed in business casual attire which seems to suggest they may be professionals or students at a professional event. \n\nIf you have any further questions, feel free to ask!",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[1],
      ],
    },
    {
      "role": "user",
      "parts": [
        "tell everything in this video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Certainly! Here's a detailed description of what's visible in the video:\n\n**Overall Scene:**\n\nThe video appears to be recorded in a room during what looks like a competition or gathering of some sort, possibly a hackathon. The camera is focused on a man in the foreground, but the background reveals many other people and activities.\n\n**Key Elements:**\n\n*   **People:**\n    *   **Man in Focus:** The primary subject of the video is a man with dark hair, a beard, and a mustache. He wears a black suit jacket and a light-colored shirt. He is initially facing the camera directly, then waves with his hand, then gives a thumbs up, and then speaks while looking at the camera.\n    *   **Other Participants:** In the background, there are many other people seated at tables, most of whom are facing laptops.\n        *   One woman is seated to the left of the man in focus in the foreground. She is wearing a dark suit jacket.\n        *   One man is standing in the right side of the frame and leans over in front of the camera. He is wearing a dark jacket. \n*   **Setting:**\n    *   **Room:** The setting appears to be a large room, likely a classroom or conference hall.\n    *   **Tables and Chairs:** The room is furnished with multiple rows of tables and chairs, arranged as if for a group activity or competition.\n    *   **Banners:** Two banners are visible on the walls. The one on the left reads, \"Hackathon Gyanotsav 2025\" and has a logo. The one on the right has a company logo and mentions \"Welcome to ...\". \n    *   **Decorations:** There are decorative ribbons hanging from the ceiling.\n    *   **Laptop Computers:** Most of the people at the tables have laptops in front of them.\n    *   **Lighting:** The room is lit by overhead lighting with bright highlights suggesting strong lighting sources. \n\n**Actions:**\n\n*   **[00:00-00:01]** The man in the foreground faces the camera directly and smiles. He appears to be looking at the camera.\n*   **[00:01]** The man waves his hand at the camera.\n*   **[00:02]** The man gives two thumbs up and looks directly at the camera. \n*   **[00:03-00:06]** The man continues speaking and gesturing. Another man appears in front of the camera for a moment. The other individuals are sitting at desks looking at their laptops.\n\n**Additional Observations**\n*  The number of people at desks, and the laptops being used point to some sort of team activity, competition or project.\n*  The banners indicate it is part of a \"Hackathon\" event.\n*  The man in focus appears to be the person who is addressing the audience.\n\nIf you have more questions about this video, feel free to ask!",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[2],
      ],
    },
    {
      "role": "user",
      "parts": [
        "tell me everything that was in video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here is a detailed breakdown of what's visible in the video:\n\n**Overall Scene:**\nThe video appears to be taken in a room that is set up for some sort of event, likely a competition. Many people are working at desks, possibly on computers. The camera is primarily focused on one individual, but it shows other participants in the background. \n\n**Key Elements:**\n\n*   **People:**\n    *   **Man in Focus:** The main subject of the video is a man with dark hair, a beard, and a mustache. He is wearing a dark suit jacket, a light-colored collared shirt, and glasses.\n    *   **Other Participants:** In the background, numerous other people are seated at tables working. Most are facing down at laptop computers. These people are also wearing dark suits.\n*   **Setting:**\n    *   **Room:** The setting looks to be a large room, perhaps a classroom or a conference hall.\n    *   **Tables and Chairs:** There are rows of tables with chairs, arranged for group work or a competition setting.\n    *   **Banners:** There are two banners visible. The left banner says \"Hackathon Gyanotsav 2025.\" The right banner is partially visible, but has the text \"Welcome to...\" written on it.\n    *   **Decorations:** There are some colorful ribbons hanging in the background.\n    *   **Laptop Computers:** Most of the people in the room appear to be using laptop computers.\n\n**Actions:**\n*   **[00:00-00:04]** The man in focus is primarily looking at the camera while talking. He touches his chin.\n*  The other participants in the room are sitting at tables looking at their laptops.\n\n**Additional Observations:**\n\n* The banners and the number of participants working on laptops suggest that this might be a coding or technology-related event, most likely a hackathon.\n* The people in the room are all wearing similar dark business attire suggesting it is a business or academic event.\n* The man in focus seems to be speaking to the camera, possibly to be part of some video coverage of the event.\n\nPlease let me know if there is anything else you would like to know!\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[3],
      ],
    },
    {
      "role": "user",
      "parts": [
        "tell me everything video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's a breakdown of what's happening in the video:\n\n**Overall Scene:**\n\nThe video appears to be a recording from a handheld camera, possibly a phone, moving through an event space. It seems like the camera is being held at a low angle, and is recording a person moving within the space. The camera is following a person. \n\n**Key Elements:**\n\n*   **People:** \n    *   **Man with Camera:** The person holding the camera is a man with dark hair, a beard, and a mustache. He is wearing a black suit jacket and a light colored shirt. \n    *   **Other Individuals:** There are several other people visible at various points:\n        *  One man appears in a gray jacket.\n        *  One woman appears to be wearing a sweater.\n        *  Another man is briefly visible wearing a dark jacket. \n*   **Setting:**\n    *   **Room:** The room appears to be a large open space, possibly a conference room or a hall with a flat ceiling. \n    *   **Tables:** There are tables visible in the background. \n    *   **Lighting:** The lighting is bright, likely from overhead lights.\n    *  **Air Conditioning Vent:** There is an air conditioning vent visible on the ceiling.\n\n**Actions:**\n*   **[00:00-00:07]** The man holding the camera is moving while recording. The camera seems to be moving to the right, then back to the left. The camera is pointed at the faces of those he passes. \n*   The other people appear to be walking or standing around. The person in gray appears to be standing still. The other individuals appear in the background.\n\n**Additional Observations:**\n*  The video suggests a casual recording being made of the participants at some event. \n*  The motion of the camera and the view of the surroundings indicate that the camera operator is walking around in the space.\n\nIf you have any more questions about the video, feel free to ask.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[4],
      ],
    },
    {
      "role": "user",
      "parts": [
        "analsye the video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Certainly! Here's an analysis of what's visible in the video:\n\n**Overall Scene:**\n\nThe video appears to be a recording from a handheld camera, possibly a phone, capturing a group of three people indoors. The camera is held at a slightly low angle, focusing primarily on their faces.\n\n**Key Elements:**\n\n*   **People:**\n    *   **Man in Front (Camera Holder):** The person holding the camera is a man with dark hair, a beard, and a mustache. He is wearing a black suit jacket and a light collared shirt, and a blue lanyard around his neck. He looks directly at the camera and smiles at various points.\n    *   **Man in the Middle:** A man is standing directly behind the camera holder. He is wearing a dark gray jacket, and looks at the camera. \n    *   **Woman on the Right:** A woman stands on the right side of the frame. She is wearing a dark sweater with a pattern on it. She is initially looking at the camera, and is holding an object, likely a camera. She later puts on a pair of headphones.\n*   **Setting:**\n    *   **Room:** The setting appears to be indoors, possibly an office or a room in a conference area.\n    *   **Ceiling:** The ceiling is white, and an air conditioning vent is visible.\n    *   **Background:**  The background is simple, not showing much detail other than a light-colored wall.\n\n**Actions:**\n\n*   **[00:00-00:05]** The camera holder is moving the camera slightly. The camera points at the faces of those in the video.\n*   **[00:00]** All three people are looking at the camera.\n*   **[00:01]** The camera holder moves the camera downward slightly.\n*    **[00:02]** The camera holder holds up his open hand to the camera.\n*   **[00:04-00:05]** The woman is holding an object and then puts on a pair of headphones.\n\n**Additional Observations:**\n\n*   The camera angle and movement make it clear that the video is recorded with a handheld device.\n*   The three people seem to be posing or interacting with the camera directly, suggesting a casual and informal moment.\n\nPlease let me know if you have any more questions about this video.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[5],
      ],
    },
    {
      "role": "user",
      "parts": [
        "describe everting in video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's a detailed description of the video:\n\n**Overall Scene:**\n\nThe video appears to be recorded indoors, likely at an event or competition. The camera is focused on two people in the foreground, but the background reveals a larger group of people.\n\n**Key Elements:**\n\n*   **People:**\n    *   **Woman on the Left:** There is a woman on the left side of the frame. She has dark hair, glasses, and she is wearing a black blazer with a blue lanyard.\n    *   **Man on the Right:** The main subject of the video is a man with dark hair, a beard, and a mustache. He is wearing a black blazer with a blue lanyard and a light colored shirt.\n    *   **Other Participants:**  In the background, there are several other people, some of whom appear to be seated at tables.\n*   **Setting:**\n    *   **Room:** The setting appears to be a large room, possibly a conference hall or a classroom, and the ceiling appears to be a flat white.\n    *  **Tables and Chairs:** There are tables and chairs visible in the background.\n    *   **Lanyards** The participants are all wearing lanyards that are the same color. The lanyard is blue with white writing.\n    *   **Decorations:** There are decorations such as banners and ribbons on the walls and ceilings. \n    *   **Air Conditioner Vent:** There is an air conditioner vent visible on the ceiling.\n\n**Actions:**\n*   **[00:00]** The two people are looking at the camera.\n*   **[00:01]** The man on the right waves his hand, while the woman on the left does not move.\n*   **[00:02]** The man on the right turns toward the right, and the camera moves slightly.\n\n**Additional Observations:**\n*  The video appears to be a casual recording or a short clip, possibly filmed as part of event coverage.\n*  The lanyards, banners, and ribbons suggest some kind of organized event, and possibly a competition.\n\nIf you have any more questions about the video, feel free to ask!\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[6],
      ],
    },
    {
      "role": "user",
      "parts": [
        "tell everything in this video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's a detailed description of everything visible in the video:\n\n**Overall Scene:**\n\nThe video appears to be recorded indoors, in a space that seems to be set up for a conference or event. The camera is focused on a person who appears to be speaking, while other people are moving in the background.\n\n**Key Elements:**\n\n*   **People:**\n    *   **Man in Focus:** The main subject of the video is a man with dark hair, a beard, and a mustache. He is wearing a black suit jacket, a light-colored shirt, and a blue lanyard. He appears to be speaking and gesturing at different moments.\n    *   **Other People:** There are several other people visible in the background:\n        *   One person is wearing a dark coat.\n        *   Another individual is wearing a light-colored furry jacket, and a dark and white hat.\n*   **Setting:**\n    *   **Room:** The room is a light colored space with light colored walls.\n    *   **Banners:** There is a banner on the wall to the left of the frame that says \"Hackathon Gyanotsav 2025\".\n    *   **Tables and Chairs:** There are tables and chairs in the background, possibly set up for group work or conference seating.\n    *   **Lighting:** The room is brightly lit, likely by overhead lights. There is a bright light glare visible.\n\n**Actions:**\n\n*   **[00:00-00:01]** The man in focus is looking toward the camera.\n*   **[00:01]** The man waves to the camera.\n*   **[00:01-00:02]** The man appears to be speaking. The camera is mostly still during this time.\n\n**Additional Observations:**\n\n*   The presence of the \"Hackathon\" banner suggests the event is related to coding, technology or computer programming.\n*   The casual movements of the people indicate that it is a informal event, and the people are moving around.\n*   The man in focus seems to be the one that the camera is following, and the recording seems to be focusing on his actions.\n\nIf you have more questions, feel free to ask.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[7],
      ],
    },
    {
      "role": "user",
      "parts": [
        "tell me everything about video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Certainly! Here's a detailed description of what's visible in the video:\n\n**Overall Scene:**\n\nThe video appears to be recorded in a large indoor space, possibly a classroom or event hall. The camera is focused on two individuals in the foreground, but the background reveals a larger group of people engaged in what seems like a collaborative or event-based activity.\n\n**Key Elements:**\n\n*   **People:**\n    *   **Man in the Foreground:** The main focus is a man with dark hair, a beard, and a mustache, wearing a black suit jacket and light-colored shirt. He is directly in front of the camera.\n    *   **Man in the Background:** Standing behind the man in the foreground is another man, with dark hair, a beard and a mustache. He is wearing a light blue sweatshirt with \"NY 09\" on it. \n    *   **Other Individuals:** There are several other people in the background, mostly seated at tables and working.\n*   **Setting:**\n    *   **Room:** The space looks like an event or conference room with a flat, white ceiling.\n    *   **Tables and Chairs:** There are multiple tables and chairs, which appear to be arranged for a group activity or competition.\n    *   **Air Conditioner Vent:** There is an air conditioner vent visible in the top of the frame.\n    *   **Lighting:** The room is brightly lit with what appears to be overhead lights.\n    *   **Water Bottle:** There is a purple water bottle visible on the table in the left side of the frame.\n\n**Actions:**\n\n*   **[00:00]** The man in the foreground is looking directly at the camera with a neutral expression. The man in the background is looking slightly down.\n*   **[00:01]** The man in the foreground raises his right arm and waves to the camera, and the man in the background is moving slightly.\n*   **[00:02]** The man in the foreground smiles at the camera and the man in the background is still moving slightly.\n*   **[00:03]** The man in the background raises his right arm and waves to the camera. The man in the foreground is still smiling.\n\n**Additional Observations:**\n\n*  The number of people at the tables suggests that it is a group activity or event, and perhaps an event with teams of participants.\n*  The overall mood of the video seems friendly and casual, and the participants appear to be in a good mood.\n\nIf you have any more questions about this video, feel free to ask!\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[8],
      ],
    },
    {
      "role": "user",
      "parts": [
        "describe everything video",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's a comprehensive description of the video:\n\n**Overall Scene:**\n\nThe video appears to be a handheld recording taken in an indoor space, likely a conference or event hall. The camera moves around and focuses on several individuals. The overall setting suggests an event or a competition.\n\n**Key Elements:**\n\n*   **People:**\n    *   **Man with Camera:** The individual holding the camera is a man with dark hair, a beard and a mustache, wearing a dark suit jacket, and light collared shirt. He is the main subject of the video.\n    *   **Other Participants:** There are multiple other individuals visible:\n        *   One woman wearing a black suit jacket.\n        *   Several people are seated at tables in the background, some with laptops in front of them.\n        *  One man wearing a black and white striped shirt, and a black jacket. \n*   **Setting:**\n    *   **Room:** The room has a light colored walls, with a white flat ceiling, and visible ceiling fans and vents.\n    *   **Banners:**  A banner is partially visible in the left side of the frame.\n    *   **Tables and Chairs:** There are tables and chairs arranged as if for group work or a conference.\n    *   **Lighting:** The lighting is bright, suggesting overhead lights.\n\n**Actions:**\n\n*   **[00:00-00:01]** The man holding the camera is in the frame, and then moves the camera to look at the table in front of him.\n*   **[00:01-00:03]** The camera focuses on the man holding the camera, and he makes a face.\n*   **[00:03-00:05]** The camera moves suddenly upward, to take a selfie type shot.\n*    **[00:05-00:09]** The man with the camera remains visible, and the camera adjusts slightly, and the man smiles at the camera. He then is standing next to another man who smiles.\n\n**Additional Observations:**\n*   The lanyards and suits suggest it may be an event for professionals or students.\n*   The number of people and the setting suggests that the video is from a organized event.\n\nLet me know if you have any more questions.\n",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)