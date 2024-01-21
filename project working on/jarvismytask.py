import datetime
import pyttsx3
import speech_recognition as sr
import time

# Set up text-to-speech engine
engine = pyttsx3.init()

# Define a function to speak the provided text
def speak(text):
    engine.setProperty('rate', 150)  # You can adjust the speaking rate
    engine.say(text)
    engine.runAndWait()

# Get today's date and time in India
def get_current_datetime():
    current_datetime = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    return current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Initialize the schedule list
schedule = []

# Function to add a task to the schedule
def add_task():
    speak("Sure, please provide the task details.")
    task = get_voice_input()
    speak("When would you like to schedule it? Please provide the time and date.")
    time, date = get_time_and_date_from_voice()
    schedule.append((task, time, date))
    speak("Task added successfully.")

# Function to retrieve today's schedule
def get_today_schedule():
    today = datetime.date.today().strftime("%Y-%m-%d")
    today_schedule = [task for task, time, date in schedule if date.startswith(today)]
    if today_schedule:
        speak("Here is your schedule for today:")
        for task in today_schedule:
            speak(task)
    else:
        speak("You don't have any tasks scheduled for today.")

# Function to capture voice commands
def listen_for_command():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio).lower()
        print("Command:", command)
        return command
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that. Can you please repeat?")
        return ""
    except sr.RequestError:
        print("Sorry, I'm currently unavailable. Please try again later.")
        return ""

# Function to capture voice input
def get_voice_input():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("Input:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that. Can you please repeat?")
        return ""
    except sr.RequestError:
        print("Sorry, I'm currently unavailable. Please try again later.")
        return ""

# Function to capture time and date from voice input
def get_time_and_date_from_voice():
    time = None
    date = None

    while not time:
        speak("Please say the time.")
        time = listen_for_command()

    while not date:
        speak("Please say the date.")
        date = listen_for_command()

    return time, date

# Function to check and remind tasks
def check_reminders():
    current_time = get_current_datetime().split(" ")[1]
    today = get_current_datetime().split(" ")[0]

    for task, time, date in schedule:
        if time == current_time and date == today:
            speak("Reminder: " + task)

# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()

# Main program loop
while True:
    command = listen_for_command()

    if "jarvis" in command:
        speak("How can I assist you?")

        command = listen_for_command()

        if "schedule" in command:
            add_task()

        elif "today" in command:
            get_today_schedule()

        elif "exit" in command:
            speak("Goodbye!")
            break

    # Check for reminders every minute
    check_reminders()
    time.sleep(666660)
